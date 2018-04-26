import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# まずどれくらいの時間回すか
# 秒数
T = 20.
# ステップ数
N = 4000
# 間隔
dt = T/float(N)
# 時間の配列を作る
t = np.arange(0.0, T, T/float(N))
count = -1
b = 1. / 6.

# おもりに関する初期条件
r1 = 2.
r2 = 1.
r3 = 1.

m1 = 2.
m2 = 1.
m3 = 1.
M = m1 + m2 + m3

g = 9.8

# おもりの位置に関する初期条件
theta1_0 = (np.pi / 180.) * 60.
theta2_0 = (np.pi / 180.) * 0.
theta3_0 = (np.pi / 180.) * 0.

# 角度をN行3列の配列に格納する（各行に角度の情報を入れていく）
theta = np.empty([N, 3])
# 初めの角度を初めの行に入れる
theta[0] = np.array([theta1_0, theta2_0, theta3_0])
# thetaの一階微分の配列を作る
omega = np.empty([N, 3])
omega[0] = np.array([0., 0., 0.])


# 二次元直交座標に直す用の配列
x = np.empty([N, 3])
y = np.empty([N, 3])
E = np.empty(N)


# r, thetaから直交座標に変換するような関数を作る．
def carx(r, angle):
    return r * np.sin(angle)


def cary(r, angle):
    return r * np.cos(angle)


# さっき作った関数を利用してthetaを入れるだけで3つの質点の座標を作ってもらう
def carxx(angle):
    xx = np.array([0., 0., 0.])
    xx[0] = carx(r1, angle[0])
    xx[1] = carx(r1, angle[0]) + carx(r2, angle[1])
    xx[2] = carx(r1, angle[0]) + carx(r2, angle[1]) + carx(r3, angle[2])
    return xx


def caryy(angle):
    yy = np.array([0., 0., 0.])
    yy[0] = cary(r1, angle[0])
    yy[1] = cary(r1, angle[0]) + cary(r2, angle[1])
    yy[2] = cary(r1, angle[0]) + cary(r2, angle[1]) + cary(r3, angle[2])
    return yy


# 左辺の奴にかかってる行列を作る
def I1(angle):
    I = np.array([[1., (m2 + m3) * np.cos(angle[0] - angle[1]) * r2 / (r1 * M),
                   m3 * np.cos(angle[0] - angle[2]) * r3 / (r1 * M)],
                  [r1 * np.cos(angle[1] - angle[0]) / r2, 1.,
                   m3 * np.cos(angle[1] - angle[2]) * r3 / (r2 * (m2 + m3))],
                  [r1 * np.cos(angle[2] - angle[0]) / r3,
                   r2 * np.cos(angle[2] - angle[1]) / r3, 1.]])
    return I


# 右辺第一項の行列
def I2(angle):
    I = np.array([[0., -(m2 + m3) * np.sin(angle[0] - angle[1]) * r2 / (r1 * M),
                   -m3 * np.sin(angle[0] - angle[2]) * r3 / (r1 * M)],
                  [-r1 * np.sin(angle[1] - angle[0]) / r2, 0.,
                   -m3 * np.sin(angle[1] - angle[2]) * r3 / (r2 * (m2 + m3))],
                  [-r1 * np.sin(angle[2] - angle[0]) / r3,
                   -r2 * np.sin(angle[2] - angle[1]) / r3, 0.]])
    return I


# ポテンシャルの行列
def I3(angle):
    I = np.array([[g * np.sin(angle[0]) / r1],
                  [g * np.sin(angle[1]) / r2],
                  [g * np.sin(angle[2]) / r3]])
    return I


# thetaの二回微分に関する微分方程式を返す
def oed(angle, dangle):
    # 左辺の行列の逆行列を作る．1行でできる．すごい．
    Iinv = np.linalg.inv(I1(angle))
    # 行列の掛け算です
    Iv = np.dot(Iinv, I2(angle))
    # thetaの一階微分の2乗の行列
    dd = np.array([[dangle[0] ** 2],
                   [dangle[1] ** 2],
                   [dangle[2] ** 2]])
    Ivd = np.dot(Iv, dd)
    U = np.dot(Iinv, I3(angle))
    return Ivd + U


# 最後ルンゲクッタを書きやすくするために定義
def runge_kutta(a, kg1, kg2, kg3, kg4):
    k1 = kg1 * np.array([b * dt])
    k2 = kg2 * np.array([2. * b * dt])
    k3 = kg3 * np.array([2. * b * dt])
    k4 = kg4 * np.array([b * dt])
    return a + k1 + k2 + k3 + k4


def Um(theta, omega):
    T1 = m1 * (r1 * omega[0]) ** 2 / 2.
    T2 = m2 * (
    (r1 * omega[0]) ** 2 + (r2 * omega[1]) ** 2 + 2. * r1 * r2 * omega[0] * omega[1] * np.cos(theta[0] - theta[1])) / 2.
    T3 = m3 * ((r1 * omega[0]) ** 2 + (r2 * omega[1]) ** 2 + (r3 * omega[2]) ** 2
               + 2. * r1 * r2 * omega[0] * omega[1] * np.cos(theta[0] - theta[1])
               + 2. * r2 * r3 * omega[1] * omega[2] * np.cos(theta[1] - theta[2])
               + 2. * r3 * r1 * omega[2] * omega[0] * np.cos(theta[2] - theta[0])) /2.
    U1 = m1 * g * r1 * np.cos(theta[0])
    U2 = m2 * g * (r1 * np.cos(theta[0]) + r2 * np.cos(theta[1]))
    U3 = m3 * g * (r1 * np.cos(theta[0]) + r2 * np.cos(theta[1]) + r3 * np.cos(theta[2]))
    return T1 + T2 + T3 + U1 + U2 + U3


E[0] = Um(theta[0], omega[0])

for i in range(N - 1):
    t[i + 1] = t[i] + dt
    # ルンゲクッタで解く
    ko1 = oed(theta[i], omega[i])
    kt1 = omega[i]

    ko2 = oed(theta[i] + kt1 * np.array(dt / 2.),
              omega[i] + ko1.ravel() * np.array(dt / 2.))
    kt2 = omega[i] + ko1.ravel() * np.array(dt / 2.)

    ko3 = oed(theta[i] + kt2 * np.array(dt / 2.),
              omega[i] + ko2.ravel() * np.array(dt / 2.))
    kt3 = omega[i] + ko2.ravel() * np.array(dt / 2.)

    ko4 = oed(theta[i] + kt3 * np.array(dt),
              omega[i] + ko3.ravel() * np.array(dt))
    kt4 = omega[i] + ko3.ravel() * np.array(dt)

    omega[i + 1] = runge_kutta(omega[i], ko1.ravel(),
                               ko2.ravel(), ko3.ravel(), ko4.ravel())
    theta[i + 1] = runge_kutta(theta[i], kt1, kt2, kt3, kt4)

    # x, y座標に変換
    x[i + 1] = carxx(theta[i + 1])
    y[i + 1] = caryy(theta[i + 1])
    E[i + 1] = Um(theta[i], omega[i])
    print(E[i+1])

plt.title("Energy")
plt.plot(t, E)
plt.grid()
plt.show()

# 質点1，2，3の座標に分ける
x1 = x[:, 0]
x2 = x[:, 1]
x3 = x[:, 2]

y1 = y[:, 0]
y2 = y[:, 1]
y3 = y[:, 2]

# こっからアニメーション．ようわからん
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5., 5.), ylim=(-5., 5.))
ax.grid()

line, = ax.plot([], [], 'o-', lw=3)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=1.8, blit=False, init_func=init)

ani.save('triple_pendulam.gif', writer="ffmpeg")
plt.show()
