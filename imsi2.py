def _full_mass_matrix_from_qM(self) -> np.ndarray:
    """
    MuJoCo의 qM(하삼각 1D 버퍼)를 full mass matrix (nv x nv)로 복원.
    mujoco.mj_fullM을 파이썬으로 구현한 버전이라고 보면 됨.
    """
    model, data = self.model, self.data
    nv = model.nv

    # qM: 길이 nv*(nv+1)/2짜리 1D 버퍼
    qM = np.array(data.qM, copy=True).ravel()
    assert qM.size == nv * (nv + 1) // 2, "qM 길이가 nv*(nv+1)/2와 다릅니다."

    M = np.zeros((nv, nv), dtype=np.float64)

    idx = 0
    for i in range(nv):
        for j in range(i + 1):  # j <= i
            v = qM[idx]
            M[i, j] = v
            M[j, i] = v
            idx += 1

    return M

def _dynamics_step(self, q, qdot, tau):
    model, data = self.model, self.data

    # 1) 현재 상태 세팅
    data.qpos[:] = q
    data.qvel[:] = qdot
    # robosuite / mujoco-py 래퍼의 forward: mj_forward와 동일한 역할
    self.sim.forward()

    # 2) full mass matrix M(q) 구성 (mujoco.mj_fullM 대신 파이썬 구현 사용)
    M = self._full_mass_matrix_from_qM()

    # 3) bias term (코리올리, 중력 등) 가져오기
    qfrc_bias = np.array(data.qfrc_bias, copy=True)

    # 4) qddot = M^{-1} (tau - qfrc_bias)
    try:
        qddot = np.linalg.solve(M, tau - qfrc_bias)
    except np.linalg.LinAlgError:
        # M이 singular하면 pseudo-inverse로 fallback
        qddot = np.linalg.pinv(M) @ (tau - qfrc_bias)

    # 5) 간단한 Euler integration (MPPI internal rollout용이라 이정도면 충분)
    q_next = q + qdot * self.dt
    qdot_next = qdot + qddot * self.dt

    return q_next, qdot_next
