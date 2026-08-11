import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# 1. SDOF 시스템 및 하중 조건 설정 (고유주기 1초 기준)
m = 1.0                         # Mass (kg)
k = 4 * (3.1415926535) ** 2     # Stiffness (N/m) ≈ 39.478 (고유주기 1초 맞춤)
damping_ratio = 0.05            # 감쇠비 5% 표준 적용
c = 2 * damping_ratio * (m * k) ** 0.5  # 감쇠계수 계산 (c = 2 * zeta * sqrt(m*k))

# 시간 설정
dt = 0.01     # Time step (sec)
T_total = 20.0 # Total simulation time (sec)
t = np.arange(0, T_total, dt)
N = len(t)

# 임의의 강제 진동 하중 P(t) 생성
P_t = 10 * np.sin(2 * t) + 5 * np.cos(5 * t) + np.random.normal(0, 1, N)

# 2. 시간이력 해석법 (Newmark-beta 평균가속도법)
gamma = 0.5
beta = 0.25

u_th = np.zeros(N)
v_th = np.zeros(N)
a_th = np.zeros(N)

a_th[0] = (P_t[0] - c * v_th[0] - k * u_th[0]) / m
k_hat = k + (gamma / (beta * dt)) * c + (1 / (beta * dt**2)) * m

for i in range(0, N - 1):
    dP = P_t[i+1] - P_t[i]
    A = (1 / (beta * dt)) * m + (gamma / beta) * c
    B = (1 / (2 * beta)) * m + dt * (gamma / (2 * beta) - 1) * c
    
    dP_hat = dP + A * v_th[i] + B * a_th[i]
    du = dP_hat / k_hat
    dv = (gamma / (beta * dt)) * du - (gamma / beta) * v_th[i] + dt * (1 - gamma / (2 * beta)) * a_th[i]
    da = (1 / (beta * dt**2)) * du - (1 / (beta * dt)) * v_th[i] - (1 / (2 * beta)) * a_th[i]
    
    u_th[i+1] = u_th[i] + du
    v_th[i+1] = v_th[i] + dv
    a_th[i+1] = a_th[i] + da

# 3. 주파수영역 해석법 (FFT & 전달함수)
P_freq = fft(P_t)
freqs = fftfreq(N, dt)
omega = 2 * np.pi * freqs

H_w = 1.0 / ((k - m * omega**2) + 1j * (c * omega))
U_freq = P_freq * H_w
u_fd = np.real(ifft(U_freq))

# 4. 결과 시각화 및 비교 Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, P_t, 'k-', label='Input Force P(t)')
plt.title('Arbitrary Dynamic Force P(t)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, u_th, 'b-', linewidth=1.5, label='Time History (Newmark-beta)')
plt.plot(t, u_fd, 'r--', linewidth=1.5, label='Frequency Domain (FFT)')
plt.title('SDOF Displacement Response Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Displacement u(t) (m)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('result_plot.png', dpi=300)
plt.show()