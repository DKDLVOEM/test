class JointTorqueMujocoMPPIController:
    """Joint-torque MPPI using MuJoCo dynamics + FK for EEF tracking (robosuite / mujoco-py compatible)."""

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dt: float,
        temperature: float,
        weights: dict,
        d_far: float,
        d_close: float,
        sim,
        ee_body_name: str,
        tau_kp: float,
        tau_kd: float,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature
        self.w_goal = weights["w_goal"]
        self.w_ref = weights["w_ref"]
        self.w_tau = weights["w_tau"]
        self.w_obs = weights["w_obs"]
        self.d_far = d_far
        self.d_close = d_close

        # robosuite / mujoco-py MjSim 객체 그대로 저장
        self.sim = sim
        self.model = getattr(sim, "model", None)
        self.data = getattr(sim, "data", None)
        if self.model is None or self.data is None:
            raise ValueError("sim에서 model / data를 가져오지 못했습니다 (robosuite MjSim 예상).")

        # EE 이름과 타입(사이트인지, 바디인지)만 저장 (id 사용 X)
        self.ee_name, self.ee_is_site = self._resolve_ee(self.model, ee_body_name)

        self.tau_kp = tau_kp
        self.tau_kd = tau_kd

    # --------------------------------------------------------
    # 외부 인터페이스: MPPI 계획
    # --------------------------------------------------------
    def plan(
        self,
        q0: np.ndarray,
        qdot0: np.ndarray,
        A_ref: np.ndarray,
        p_obj: np.ndarray,
        v_obj: np.ndarray,
        gripper_cmd_ref: float,
    ) -> np.ndarray:
        """
        Returns:
            delta_chunk: (K,7) delta-EEF+gripper sequence mapped from torque plan (first element used for env.step).
        """
        nv = self.model.nv
        H_ref = A_ref.shape[0]
        K = min(self.horizon, H_ref)

        # 현재 EE 위치에서 목표 물체까지 거리 → phase 결정
        p_ee0 = self._get_ee_pos(q0, qdot0)
        d = np.linalg.norm(p_ee0 - p_obj)
        phase = self._compute_phase(d, gripper_cmd_ref)
        w_goal, w_ref, w_tau, noise_scale = self._phase_weights_noise(phase)

        # grasp phase면 그냥 pi0 그대로 따라가기
        if phase == "grasp":
            return A_ref[:K]

        # reference eef 경로, reference torque 계산
        p_ref = self._build_eef_ref(p_ee0, A_ref[:K, :3])
        tau_ref = self._build_tau_ref(q0, qdot0, A_ref[:K])

        # 뉴이즈 샘플
        sigma_vec = np.ones(nv, dtype=np.float32) * self.dt * noise_scale
        eps = np.random.normal(scale=sigma_vec, size=(self.num_samples, K, nv))
        costs = np.zeros(self.num_samples, dtype=np.float32)

        # --------- 샘플별 rollout (내부 시뮬레이션) ----------
        for j in range(self.num_samples):
            q = q0.copy()
            qdot = qdot0.copy()
            J = 0.0
            for k in range(K):
                tau_k = tau_ref[k] + eps[j, k]
                q, qdot = self._dynamics_step(q, qdot, tau_k)
                t_pred = (k + 1) * self.dt
                p_obj_pred = p_obj + v_obj * t_pred
                p_ee = self._get_ee_pos(q, qdot)

                L_goal = w_goal * np.sum((p_ee - p_obj_pred) ** 2)
                L_ref = w_ref * np.sum((p_ee - p_ref[k]) ** 2)
                L_tau = w_tau * np.sum(tau_k ** 2)
                J += L_goal + L_ref + L_tau
            costs[j] = J

        # --------- MPPI weight 계산 ----------
        J_min = np.min(costs)
        weights = np.exp(-(costs - J_min) / max(self.temperature, 1e-6))
        weights = weights / (np.sum(weights) + 1e-8)

        # 참고 torque + 가중 평균 noise로 최종 torque trajectory 생성
        tau_new = tau_ref + np.sum(weights[:, None, None] * eps, axis=0)  # (K, nv)

        # --------- 최종 tau_new로 deterministic rollout → delta-EEF chunk ----------
        delta_chunk = np.zeros((K, 7), dtype=np.float32)
        q = q0.copy()
        qdot = qdot0.copy()
        p_prev = p_ee0.copy()
        for k in range(K):
            q, qdot = self._dynamics_step(q, qdot, tau_new[k])
            p_curr = self._get_ee_pos(q, qdot)
            delta_pos = p_curr - p_prev
            delta_ori = A_ref[k][3:6]  # pi0가 준 orientation delta 유지
            gripper_cmd = A_ref[k][-1]
            delta_chunk[k] = np.concatenate([delta_pos, delta_ori, [gripper_cmd]])
            p_prev = p_curr

        return delta_chunk

    # --------------------------------------------------------
    # 내부 dynamics / kinematics 유틸
    # --------------------------------------------------------
    def _dynamics_step(self, q, qdot, tau):
        """
        단일 타임스텝 dynamics 업데이트.
        robosuite의 MjData.M (full mass matrix)와 qfrc_bias를 사용.
        """
        model, data = self.model, self.data

        # q, qdot 세팅 후 forward
        data.qpos[:] = q
        data.qvel[:] = qdot
        self._forward()

        # full mass matrix M (nv x nv)
        M = np.array(data.M, dtype=np.float64)
        M = M.reshape(model.nv, model.nv)

        # qddot = M^{-1} (tau - bias)
        qddot = np.linalg.solve(M, tau - data.qfrc_bias)

        # explicit Euler
        q_next = q + qdot * self.dt
        qdot_next = qdot + qddot * self.dt
        return q_next, qdot_next

    def _get_ee_pos(self, q, qdot):
        """
        현재 joint 상태(q, qdot)에서 EE 위치를 반환.
        robosuite의 get_site_xpos / get_body_xpos 사용.
        """
        data = self.data
        data.qpos[:] = q
        data.qvel[:] = qdot
        self._forward()

        if self.ee_is_site:
            return np.array(data.get_site_xpos(self.ee_name), dtype=np.float64)
        else:
            return np.array(data.get_body_xpos(self.ee_name), dtype=np.float64)

    def _build_tau_ref(self, q0: np.ndarray, qdot0: np.ndarray, A_ref: np.ndarray) -> np.ndarray:
        nv = self.model.nv
        K = A_ref.shape[0]
        tau_ref = np.zeros((K, nv), dtype=np.float32)
        q = q0.copy()
        qdot = qdot0.copy()
        for k in range(K):
            dp_ref = A_ref[k, :3]
            tau_k = self._eef_delta_to_tau(q, qdot, dp_ref)
            tau_ref[k] = tau_k
            q, qdot = self._dynamics_step(q, qdot, tau_k)
        return tau_ref

    def _resolve_ee(self, model, name_hint):
        """
        EE를 body/site 이름으로 resolve.
        - 우선 정확한 이름(body_name2id, site_name2id) 시도
        - 안 되면 hand / gripper / eef / name_hint 포함된 body/site fuzzy 검색
        - 그래도 없으면 root body 이름으로 fallback
        """
        # 1) direct 이름 매칭
        if name_hint:
            try:
                model.body_name2id(name_hint)
                return name_hint, False
            except Exception:
                pass
            try:
                model.site_name2id(name_hint)
                return name_hint, True
            except Exception:
                pass

        # 2) fuzzy body 검색
        name_hint_l = name_hint.lower() if name_hint else ""
        for idx in range(model.nbody):
            n = model.body_id2name(idx)
            if not n:
                continue
            nl = n.lower()
            if any(k in nl for k in ["hand", "gripper", "eef", name_hint_l]):
                return n, False

        # 3) fuzzy site 검색
        for idx in range(model.nsite):
            n = model.site_id2name(idx)
            if not n:
                continue
            nl = n.lower()
            if any(k in nl for k in ["hand", "gripper", "eef", name_hint_l]):
                return n, True

        # 4) 최종 fallback: body 0 이름
        return model.body_id2name(0), False

    def _build_eef_ref(self, p_start: np.ndarray, dp_seq: np.ndarray) -> np.ndarray:
        K = dp_seq.shape[0]
        prefs = np.zeros((K, 3), dtype=np.float32)
        acc = p_start.copy()
        for k in range(K):
            acc = acc + dp_seq[k]
            prefs[k] = acc
        return prefs

    def _eef_delta_to_tau(self, q: np.ndarray, qdot: np.ndarray, dp: np.ndarray) -> np.ndarray:
        """
        EE position delta(dp)을 joint torque로 매핑.
        robosuite의 get_*_jacp / get_*_jacr 사용.
        """
        data = self.data
        data.qpos[:] = q
        data.qvel[:] = qdot
        self._forward()

        # 현재 EE 위치
        if self.ee_is_site:
            p_cur = np.array(data.get_site_xpos(self.ee_name), dtype=np.float64)
            jacp_flat = np.array(data.get_site_jacp(self.ee_name), dtype=np.float64)
        else:
            p_cur = np.array(data.get_body_xpos(self.ee_name), dtype=np.float64)
            jacp_flat = np.array(data.get_body_jacp(self.ee_name), dtype=np.float64)

        # jacobian reshape: (3*nv,) -> (3, nv)
        jacp = jacp_flat.reshape(3, self.model.nv)

        # EE 선속도
        v_ee = jacp @ qdot

        # 목표 위치 = 현재 + dp
        p_des = p_cur + dp
        err = p_des - p_cur

        # Cartesian PD -> 힘
        F = self.tau_kp * err - self.tau_kd * v_ee

        # joint torque
        tau = jacp.T @ F
        return tau.astype(np.float32)

    # --------------------------------------------------------
    # Phase / weight / forward 유틸
    # --------------------------------------------------------
    def _compute_phase(self, d: float, g_cmd: float) -> str:
        if d <= self.d_close or g_cmd < -0.2:
            return "grasp"
        if d <= self.d_far:
            return "near"
        return "approach"

    def _phase_weights_noise(self, phase: str) -> Tuple[float, float, float, float]:
        w_goal = self.w_goal
        w_ref = self.w_ref
        w_tau = self.w_tau
        noise_scale = 1.0
        if phase == "approach":
            w_goal *= 1.5
            w_ref *= 0.5
            noise_scale *= 1.5
        elif phase == "near":
            w_goal *= 1.0
            w_ref *= 2.0
            noise_scale *= 0.5
        elif phase == "grasp":
            w_goal *= 0.5
            w_ref *= 10.0
            noise_scale *= 0.1
        return w_goal, w_ref, w_tau, noise_scale

    def _forward(self):
        """
        robosuite / mujoco-py의 sim.forward() 사용.
        (env.env.sim.forward 와 동일한 내부 호출)
        """
        self.sim.forward()
