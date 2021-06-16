import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def get_data(file):
    Vacc = []
    time_ns = []
    particles = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'Time' in lines[i].split():
                Vacc.append(lines[i].split()[-1])

            if lines[i].split():
                try:
                    float(lines[i].split()[0])
                    ans = True
                except ValueError:
                    ans = False
                if ans:
                    time_ns.append(float(lines[i].split()[0]))
                    particles.append(float(lines[i].split()[1]))
    time_ns_f = np.reshape(time_ns, (len(Vacc), int(len(time_ns)/len(Vacc))))
    particles_f = np.reshape(particles, (len(Vacc), int(len(time_ns)/len(Vacc))))
    return Vacc, time_ns_f, particles_f


def exponential(t, N0, alpha):
    return N0 * np.exp(alpha * t)


def plot_CST_data(Vacc, time, particles, title, exponential_fit_display=False, log_y_scale=False):
    alpha_ls = []
    vacc_ls = []
    sns.set_palette(sns.color_palette("inferno_r", len(Vacc)))
    for i in range(1, len(time)):
        plt.plot(time[i], particles[i], label=Vacc[i].split('=')[1].split(')')[0] + ' kV')
        vacc_ls.append(Vacc[i].split('=')[1].split(')')[0])
        plt.legend(loc='best', fancybox=False, framealpha=0.7, edgecolor='k', ncol=3)
        pars, cov = curve_fit(f=exponential, xdata=time[i][630::], ydata=particles[i][630::], p0=[0, 0],
                              bounds=(-np.inf, np.inf))
        if exponential_fit_display:
            plt.plot(time[i], exponential(time[i], *pars), linestyle='--', linewidth=2,
                     label='Fit of ' + Vacc[i].split('=')[1].split(')')[0] + ' kV')
        alpha_ls.append(pars[1])
    plt.grid(alpha=0.5, ls=':')
    plt.xlabel(r'Time (ns)', fontsize=13)
    plt.ylabel('Number of particles', fontsize=13)
    plt.title(title)
    if log_y_scale:
        plt.yscale('log')
    return vacc_ls, alpha_ls

# if __name__ == '__main__':
#     Vacc, time_ns_f, particles_f = get_data('/Users/chen/Desktop/ANL_work/WiFEL/CST/multipacting/mp_bottom_source_100ns.txt')
#     plt.plot(time_ns_f[0], particles_f[0])
#     plt.show()
