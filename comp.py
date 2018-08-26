import numpy as np
from time import clock
import matplotlib.pyplot as plt

a=-1.0
b=5.0
max = 30000
def int_f():
    return 1.6840782256556266186

def f(x):
    return np.multiply(x,np.sin(3*x))

def fp(x):
    return np.sin(3*x)+np.multiply(3*x,np.cos(3*x))

def trap(a,b,f,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    F = f(x)
    I_app = (F[0] + F[-1])*h/2.0
    for i in range(1,n):
        I_app += h*F[i]
    return I_app

def h_1(a,b,f,fp,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    F = f(x)
    FP_l = fp(x[0])
    FP_r = fp(x[-1])
    I_app = (F[0] + F[-1])*h/2.0 + (FP_l - FP_r)*h**2/12.0
    for i in range(1,n):
        I_app += h*F[i]
    return I_app

def h_1_fd(a,b,f,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    if abs(h)<4.9e-5:
        print("Round off due to h being too small.")
    F = f(x)
    FP_l = np.dot([-50, 96,-72, 32, -6],F[0:5])/(24*h)
    FP_r = np.dot([-50, 96,-72, 32, -6],np.flip(F[-5:],0))/(24*(-h))
    I_app = (F[0] + F[-1])*h/2.0 + (FP_l - FP_r)*h**2/12.0
    for i in range(1,n):
        I_app += h*F[i]
    return I_app

def simpsons(a,b,f,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    F = f(x)
    I_app = F[0] + F[-1]
    for i in range(1,n+1,2):
        I_app += 4*F[i]
    for i in range(2,n,2):
        I_app += 2*F[i]
    I_app *= h/3.0
    return I_app

def h_2(a,b,f,fp,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    F = f(x)
    FP_l = fp(x[0])
    FP_r = fp(x[-1])
    w = [7*h/15, 16*h/15, h**2/15]
    I_app = w[2]*(FP_l - FP_r)
    for i in range(1, n, 2):
        I_app += w[0]*(F[i-1] + F[i+1]) + w[1]*F[i]
    return I_app

def h_2_fd(a,b,f,n):
    x = np.linspace(a,b,n+1)
    h = (x[-1]-x[0])/n
    if abs(h)<6.3e-4:
        print("Round off due to h being too small.")
    F = f(x)
    FP_l = np.dot([-274,600,-600,400,-150,24],F[0:6])/(120*h)
    FP_r = np.dot([-274,600,-600,400,-150,24],np.flip(F[-6:],0))/(120*(-h))
    w = [7*h/15, 16*h/15, h**2/15]
    I_app = w[2]*(FP_l - FP_r)
    for i in range(1, n, 2):
        I_app += w[0]*(F[i-1] + F[i+1]) + w[1]*F[i]
    return I_app

def findN(est, tol, func1, isHermite):
    a1 = est
    b1 = a1/10
    while (a1 - b1) >= 2:
        c = (a1 + b1) / 2
        c = int(np.ceil(c))
        
        if isHermite:
            fc = func1(a, b, f, fp, c)
        else:
            fc = func1(a, b, f, c)
            
        if abs(int_f() - fc) < tol:
            a1 = c          
        else:
            b1 = c
        
    while abs(int_f() - fc) > tol:
        c += 1
        if isHermite:
            fc = func1(a, b, f, fp, c)
        else:
            fc = func1(a, b, f, c)
    return int(np.ceil(c))

N1 = [286,903,2854,28539,285381,2845697]
Tol = [0.5e-03,0.5e-04,0.5e-05,0.5e-07,0.5e-09,0.5e-11,0.5e-15]
print('   tol        n     Hermite_1      Hermite_1_FD   Trapezoid')
k = 0
for n in N1:
    x = np.linspace(a,b,n+1)
    print('%1.1e   %7d   %1.6e   %1.6e   %1.6e' \
          % (Tol[k],n,abs(int_f() - h_1(a,b,f,fp,n)),\
             abs(int_f() - h_1_fd(a,b,f,n)),abs(int_f() - trap(a,b,f,n))))
    k += 1
print('\n')

N2 = [36,64,112,354,1118,3532,182884]

print('   tol         n    Hermite_2      Hermite_2_FD   Simpsons')
k = 0
for n in N2:
    x = np.linspace(a,b,n+1)
    print('%1.1e   %7d   %1.6e   %1.6e   %1.6e' \
          % (Tol[k],n,abs(int_f() - h_2(a,b,f,fp,n)),\
             abs(int_f() - h_2_fd(a,b,f,n)),abs(int_f() - simpsons(a,b,f,n))))
    k += 1
print('\n\n')


print('   tol          n   Hermite_1           n    Hermite_1_FD       n   Trapezoid')
n1_h = findN(N2[0], Tol[0], h_1, True)
n1_hfd = findN(N2[0], Tol[0], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[0],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)),286,abs(int_f() - trap(a,b,f,286))))
n1_h = findN(N2[1], Tol[1], h_1, True)
n1_hfd = findN(N2[1], Tol[1], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[1],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
     n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)),903,abs(int_f() - trap(a,b,f,903))))
n1_h = findN(N2[2], Tol[2], h_1, True)
n1_hfd = findN(N2[2], Tol[2], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[2],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)),2854,abs(int_f() - trap(a,b,f,2854))))
n1_h = findN(N2[3], Tol[3], h_1, True)
n1_hfd = findN(N2[3], Tol[3], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[3],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)),28539,abs(int_f() - trap(a,b,f,28539))))
n1_h = findN(N2[4], Tol[4], h_1, True)
n1_hfd = findN(N2[4], Tol[4], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[4],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)), 285385,abs(int_f() - trap(a,b,f,285385))))
n1_h = findN(N2[5], Tol[5], h_1, True)
n1_hfd = findN(N2[5], Tol[5], h_1_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[5],n1_h,abs(int_f() - h_1(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_1_fd(a,b,f,n1_hfd)),2859229,abs(int_f() - trap(a,b,f,2859229))))
print('\n')

print('   tol          n   Hermite_2           n    Hermite_2_FD       n   Simpson\'s')
n1_h = findN(N2[0], Tol[0], h_2, True)
n1_hfd = findN(N2[0], Tol[0], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[0],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),36,abs(int_f() - simpsons(a,b,f,36))))
n1_h = findN(N2[0], Tol[1], h_2, True)
n1_hfd = findN(N2[0], Tol[1], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[1],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),64,abs(int_f() - simpsons(a,b,f,64))))
n1_h = findN(N2[0], Tol[2], h_2, True)
n1_hfd = findN(N2[0], Tol[2], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[2],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),112,abs(int_f() - simpsons(a,b,f,112))))
n1_h = findN(N2[1], Tol[3], h_2, True)
n1_hfd = findN(N2[1], Tol[3], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[3],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),354,abs(int_f() - simpsons(a,b,f,354))))
n1_h = findN(N2[2], Tol[4], h_2, True)
n1_hfd = findN(N2[2], Tol[4], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[4],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),1118,abs(int_f() - simpsons(a,b,f,1118))))
n1_h = findN(N2[2], Tol[5], h_2, True)
n1_hfd = findN(N2[2], Tol[5], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[5],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
      n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),3532,abs(int_f() - simpsons(a,b,f,3532))))
n1_h = findN(N2[3], Tol[6], h_2, True)
n1_hfd = findN(N2[3], Tol[6], h_2_fd, False)
print('%1.1e   %7d   %1.6e   %7d   %1.6e  %7d  %1.6e' \
      % (Tol[6],n1_h,abs(int_f() - h_2(a,b,f,fp,n1_h)),\
         n1_hfd,abs(int_f() - h_2_fd(a,b,f,n1_hfd)),249994,abs(int_f() - simpsons(a,b,f,249994))))


def get_time_trap(n):
    iters = int(np.ceil(max / n))
    if n > 2861:
        iters = 10
    start = clock()
    for i in range(iters):
        int_trap(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_simp(n):
    iters = int(np.ceil(max / n))
    if n > 61:
        iters = 10
    start = clock()
    for i in range(iters):
        int_simp(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_simp(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        simpsons(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_trap(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        trap(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_h1(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        h_1(a,b,f,fp,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_h1_fd(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        h_1_fd(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_h2(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        h_2(a,b,f,fp,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_h2_fd(n):
    iters = int(np.ceil(max / n))
    start = clock()
    for i in range(iters):
        h_2_fd(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

trapTimes = [get_time_trap(286), get_time_trap(903), get_time_trap(2854), \
             get_time_trap(28539), get_time_trap(285385), get_time_trap(2859229)]
h1Times = [get_time_h1(26),get_time_h1(45), get_time_h1(79), get_time_h1(250), \
           get_time_h1(790), get_time_h1(2497)]
h1_fdTimes = [get_time_h1_fd(31), get_time_h1_fd(38), get_time_h1_fd(82), \
              get_time_h1_fd(253), get_time_h1_fd(791), get_time_h1_fd(2497)]

simpTimes = [get_time_simp(36), get_time_simp(64), get_time_simp(112), get_time_simp(354), \
             get_time_simp(1118), get_time_simp(3532), get_time_simp(249994)]
h2Times = [get_time_h2(16), get_time_h2(22), get_time_h2(32), get_time_h2(68), get_time_h2(144), \
           get_time_h2(308), get_time_h2(1054)]
h2_fdTimes = [get_time_h2_fd(20), get_time_h2_fd(44), get_time_h2_fd(62), get_time_h2_fd(120), \
              get_time_h2_fd(228), get_time_h2_fd(434), get_time_h2_fd(1220)]

fig, ax = plt.subplots()
for i in range(0,6):
    trap, = ax.plot(Tol[i], trapTimes[i],'o', color = 'red')
    h1, = ax.plot(Tol[i], h1Times[i], 'o', color = 'green')
    h1_fd, = ax.plot(Tol[i], h1_fdTimes[i], 'o', color = 'blue')

ax.set_title('Tolerance vs. Computational Cost')
ax.set_xscale('log', base = 10)
ax.set_yscale('log', base = 10)
ax.set_xlabel('Tolerance')
ax.set_ylabel('Computation Cost')
plt.legend([trap, h1, h1_fd], ['Trapezoid', 'Hermite_1', 'Hermite_1_FD'], loc = 'upper right')
plt.show()

fig, ax = plt.subplots()
for i in range(0,7):
    simpsons, = ax.plot(Tol[i], simpTimes[i],'o', color = 'red')
    h2, = ax.plot(Tol[i], h2Times[i], 'o', color = 'green')
    h2_fd, = ax.plot(Tol[i], h2_fdTimes[i], 'o', color = 'blue')

ax.set_title('Tolerance vs. Computational Cost')
ax.set_xscale('log', base = 10)
ax.set_yscale('log', base = 10)
ax.set_xlabel('Tolerance')
ax.set_ylabel('Computation Cost')
plt.legend([simpsons, h2, h2_fd], ['Simpson\'s', 'Hermite_2', 'Hermite_2_FD'], loc = 'upper right')
plt.show()
