import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 시스템 파라미터
# ============================================================

m = 1.0                    # 질량
k = 4 * np.pi**2           # 강성
zeta = 0.05                # 감쇠비

wn = np.sqrt(k / m)        # 고유원진동수
fn = wn / (2 * np.pi)      # 고유진동수

# ============================================================
# 2. 시간
# ============================================================

dt = 0.01
t = np.arange(0, 10, dt)

# ============================================================
# 3. Linear response
# 일정한 가진이 계속된다고 가정
# → 처음과 끝의 진폭이 거의 동일
# ============================================================

A_linear = 1.0

linear_strain = A_linear * np.sin(wn * t)


# ============================================================
# 4. Nonlinear / transient response
# 초기에 큰 변형 발생 → 시간에 따라 감소
# ============================================================

# 비선형 응답이 크게 발생하는 시점
t_peak = 1.5

# 감쇠 정도
decay = 0.70

# 초기 지진성 불규칙 응답
burst = np.exp(-((t - t_peak) / 0.45)**2)

nonlinear_strain = (
    1.15 * burst * np.sin(4.5 * wn * t)
    + 0.55 * burst * np.sin(7.0 * wn * t)
)

# peak 이후 점차 감소하는 진동
after = t >= t_peak

nonlinear_strain[after] += (
    0.25
    * np.exp(-decay * (t[after] - t_peak))
    * np.sin(2.3 * wn * (t[after] - t_peak))
)

# 작은 잔류진동
np.random.seed(10)

noise_envelope = np.where(
    t < t_peak,
    0.01,
    0.035 * np.exp(-0.35 * (t - t_peak))
)

nonlinear_strain += noise_envelope * np.random.randn(len(t))


# ============================================================
# 5. Peak 값 맞추기
# Kramer 그림처럼 두 곡선의 최대값을 동일하게 조정
# ============================================================

nonlinear_strain = (
    nonlinear_strain
    / np.max(np.abs(nonlinear_strain))
    * np.max(np.abs(linear_strain))
)


# ============================================================
# 6. 그래프
# ============================================================

# ============================================================
# 6. 그래프
# ============================================================

plt.figure(figsize=(12, 5))

# 선형 응답 - 파란색
plt.plot(
    t,
    linear_strain,
    color="blue",
    linewidth=1.5,
    label="Linear Analysis"
)

# 비선형 응답 - 빨간색
plt.plot(
    t,
    nonlinear_strain,
    color="red",
    linewidth=1.2,
    label="Nonlinear Analysis"
)

# 0 기준선
plt.axhline(
    0,
    color="black",
    linewidth=0.7
)

# X축을 정확히 0초부터 10초까지 표시
plt.xlim(0, 10)

# X축 자동 여백 제거
plt.margins(x=0)

plt.xlabel("Time (sec)")
plt.ylabel("Shear Strain")
plt.title("Linear vs Nonlinear Shear Strain Time Histories")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("nonlinear_analysis_result.png", dpi=300, bbox_inches="tight")
plt.show()