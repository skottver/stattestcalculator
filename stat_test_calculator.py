import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, lognorm, norm
from scipy.optimize import minimize_scalar, brentq

class StatTestCalculator:
    def __init__(self, signal_hist=None, background_hist=None, data=None,
                 delta=None, delta_s=None, test_stat_type = 'q', mode='significance', prior='lognormal',
                 ntoys=10000, ul_grid=None, cl=0.95, verbose=True,
                 method='hybrid', syst_type='uncorrelated'):
        self.signal = np.asarray(signal_hist) if signal_hist is not None else None
        self.background = np.asarray(background_hist) if background_hist is not None else None
        self.data = np.asarray(data) if data is not None else None
        self.delta = delta
        self.delta_s = delta_s
        self.syst_type = syst_type
        self.mode = mode
        self.prior = prior
        self.ntoys = ntoys
        self.ul_grid = ul_grid
        self.cl = cl
        self.verbose = verbose
        self.method = method
        self.nbins = self._validate_histograms()
        self.test_stat_type = test_stat_type

        self._q0_dist = None
        self._q0_obs = None
        self._qmu_obs = None
        self._qmu_dists = None
        self._p_values = None

    def _validate_histograms(self):
        if self.signal is not None and self.background is not None:
            if len(self.signal) != len(self.background):
                raise ValueError(f"Signal and background histograms must have the same length. Got {len(self.signal)} and {len(self.background)}")
            return len(self.signal)
        return 0

    def import_data(self, data=None, signal_hist=None, background_hist=None):
        if signal_hist is not None:
            self.signal = np.asarray(signal_hist)
        if background_hist is not None:
            self.background = np.asarray(background_hist)
        if data is not None:
            self.data = np.asarray(data)
        self.nbins = self._validate_histograms()

    def _log_likelihood(self, mu, data, signal, background):
        expected = mu * signal + background
        expected = np.where(expected <= 0, 1e-9, expected)
        log_pmf = poisson.logpmf(data, expected)
        logL = np.sum(log_pmf)

        if self.method == 'frequent' and self.delta:
            if self.prior == 'lognormal':
                sigma = np.sqrt(np.log(1 + (self.delta ** 2)))
                scale = np.exp(-0.5 * sigma ** 2)
                if self.syst_type == 'uncorrelated':
                    syst = lognorm.rvs(sigma, scale=scale, size=self.nbins)
                else:  
                    syst = lognorm.rvs(sigma, scale=scale, size=1)
                logL += np.sum(lognorm(sigma, scale=scale).logpdf(background / (self.background * syst)))
            elif self.prior == 'gauss':
                sigma = self.delta * self.background
                logL += np.sum(norm(loc=self.background, scale=sigma).logpdf(background))
            else:
                raise ValueError(f"Unsupported prior: {self.prior}")
            
        if self.method == 'frequent' and self.delta_s:
            if self.prior == 'lognormal':
                sigma = np.sqrt(np.log(1 + (self.delta_s ** 2)))
                scale = np.exp(-0.5 * sigma ** 2)
                if self.syst_type == 'uncorrelated':
                    syst = lognorm.rvs(sigma, scale=scale, size=self.nbins)
                else:  
                    syst = lognorm.rvs(sigma, scale=scale, size=1)
                logL += np.sum(lognorm(sigma, scale=scale).logpdf(signal / (self.signal * syst)))
            elif self.prior == 'gauss':
                sigma = self.delta_s * self.signal
                logL += np.sum(norm(loc=self.signal, scale=sigma).logpdf(signal))
            else:
                raise ValueError(f"Unsupported prior: {self.prior}")

        return logL

    def _find_mu_hat(self, data, signal, background):
        def neg_ll(mu):
            return -self._log_likelihood(mu, data, signal, background)

        upper = max(10, (np.sum(data - background) / np.sum(signal)) if np.sum(signal) > 0 else 10)
        res = minimize_scalar(neg_ll, bounds=(0, upper), method='bounded')
        if not res.success:
            raise RuntimeError("Failed to find mu_hat")
        return res.x


    def _calculate_clsb(self, data=None, mu_null = 1):
        data = np.asarray(data)
        l_sb = self._log_likelihood(mu_null, data, self.signal, self.background)
        l_b = self._log_likelihood(0.0, data, self.signal, self.background)
        return -2 * (l_sb - l_b)

    def _calculate_q0(self, data=None):
        data = np.asarray(data)
        ll0 = self._log_likelihood(0.0, data, self.signal, self.background)
        mu_hat = self._find_mu_hat(data, self.signal, self.background)
        ll_hat = self._log_likelihood(mu_hat, data, self.signal, self.background)
        q0 = -2 * (ll0 - ll_hat)
        return max(0, q0)

    def _calculate_qmu(self, mu, data):
        data = np.asarray(data)
        ll_mu = self._log_likelihood(mu, data, self.signal, self.background)
        mu_hat = self._find_mu_hat(data, self.signal, self.background)
        ll_hat = self._log_likelihood(mu_hat, data, self.signal, self.background)
        return max(0, -2 * (ll_mu - ll_hat)) if mu_hat <= mu else 0

    def monte_carlo_hypotest(self, test_stat_type = None):
        test_stat_type = test_stat_type if test_stat_type else self.test_stat_type

        if self.mode == 'significance':
            if test_stat_type == 'q':
                self._q0_obs = self._calculate_q0(self.data)
            elif test_stat_type == 'clsb':
                self._q0_obs = self._calculate_clsb(self.data, mu_null = 1)
            else:
                raise ValueError(f'Unsupported test statistic: {test_stat_type}')
            q0_dist = []
            for i in range(self.ntoys):
                b_toy = self._generate_background()
                toy_data = np.random.poisson(b_toy)
                if test_stat_type == 'q':
                    q0 = self._calculate_q0(toy_data)
                elif test_stat_type == 'clsb':
                    q0 = self._calculate_clsb(toy_data, mu_null = 1)
                q0_dist.append(q0)
                if self.verbose and i % 500 == 0:
                    print(f'{i}/{self.ntoys}')
            self._q0_dist = np.array(q0_dist)
            if test_stat_type == 'q':
                p = np.mean(self._q0_dist >= self._q0_obs, dtype=np.longdouble)
            elif test_stat_type == 'clsb':
                p = np.mean(self._q0_dist <= self._q0_obs, dtype=np.longdouble)
            p = np.float64(p)
            if p == 0:
                p = 1 / (self.ntoys + 1)
            z = max(0, norm.ppf(1 - p))
            if self.verbose:
                print(f"Observed statistic: {self._q0_obs:.3f}, p-value: {p:.5f}, significance: {z:.3f}σ")
            return p, z

        elif self.mode == 'upper_limits':
            if self.ul_grid is None:
                self.ul_grid = np.linspace(0, 5, 11)
            self._qmu_obs = []
            for mu in self.ul_grid:
                if test_stat_type == 'q':
                    self._qmu_obs.append(self._calculate_qmu(mu, self.data))
                elif test_stat_type == 'clsb':
                    self._qmu_obs.append(self._calculate_clsb(self.data, mu_null = mu))
            self._p_values = []
            self._qmu_dists = []
            mu_ul = None
            for i, mu in enumerate(self.ul_grid):
                qmu_dist = []
                for _ in range(self.ntoys):
                    b_toy = self._generate_background()
                    s_toy = self._generate_signal()
                    toy_data = np.random.poisson(mu * self.signal + b_toy)
                    if test_stat_type == 'q':
                        qmu = self._calculate_qmu(mu, toy_data)
                    elif test_stat_type == 'clsb':
                        qmu = self._calculate_clsb(toy_data, mu_null = mu)
                    qmu_dist.append(qmu)
                qmu_dist = np.array(qmu_dist)
                self._qmu_dists.append(qmu_dist)
                p_mu = np.mean(qmu_dist >= self._qmu_obs[i])
                self._p_values.append(p_mu)
                if self.verbose:
                    print(f"mu={mu:.2f}: p={p_mu:.3f}")
                if p_mu <= 1 - self.cl:
                    mu_ul = mu
                    break
            if mu_ul is None and self.verbose:
                print("Upper limit not found in grid")
            elif self.verbose:
                print(f"Approx. {int(self.cl*100)}% CL upper limit: mu < {mu_ul:.2f}")
            return mu_ul
        else:
            raise ValueError("Unknown mode, choose 'significance' or 'upper_limits'")

    def _generate_background(self):
        if self.delta is not None and self.method == 'hybrid':
            if self.prior == 'lognormal':
                sigma = np.sqrt(np.log(1 + (self.delta**2)))
                scale = np.exp(-0.5 * sigma**2)
                if self.syst_type == 'correlated':
                    syst = lognorm.rvs(sigma, scale=scale, size=1)
                elif self.syst_type == 'uncorrelated':
                    syst = lognorm.rvs(sigma, scale=scale, size=self.nbins)
                else:
                    raise ValueError(f'Only "correlated" or "correlated" syst_type are available, got {self.syst_type}')
                
            elif self.prior == 'gauss':
                if self.syst_type == 'correlated':
                    syst = 1 + np.random.normal(0, self.delta, size=1)
                    syst = np.clip(syst, 1e-3, None)
                elif self.syst_type == 'uncorrelated':
                    syst = 1 + np.random.normal(0, self.delta, size=self.nbins)
                    syst = np.clip(syst, 1e-3, None)
            else:
                raise ValueError("Unknown prior: choose 'lognormal' or 'gauss'")
            
            return self.background * syst 
        else:
            return self.background

    def _generate_signal(self):
        if self.delta_s is not None and self.method == 'hybrid':
            if self.prior == 'lognormal':
                sigma = np.sqrt(np.log(1 + (self.delta_s**2)))
                scale = np.exp(-0.5 * sigma**2)
                if self.syst_type == 'correlated':
                    syst = lognorm.rvs(sigma, scale=scale, size=1)
                elif self.syst_type == 'uncorrelated':
                    syst = lognorm.rvs(sigma, scale=scale, size=self.nbins)
                else:
                    raise ValueError(f'Only "correlated" or "correlated" syst_type are available, got {self.syst_type}')
                
            elif self.prior == 'gauss':
                if self.syst_type == 'correlated':
                    syst = 1 + np.random.normal(0, self.delta_s, size=1)
                    syst = np.clip(syst, 1e-3, None)
                elif self.syst_type == 'uncorrelated':
                    syst = 1 + np.random.normal(0, self.delta_s, size=self.nbins)
                    syst = np.clip(syst, 1e-3, None)
            else:
                raise ValueError("Unknown prior: choose 'lognormal' or 'gauss'")
            return self.signal * syst 
        else:
            return self.signal
    
    def _asymptotic_significance(self,
                                 signal=None,
                                 background=None,
                                 delta=None):
        
        S = np.array(signal     if signal     is not None else self.signal_hist,     float)
        B = np.array(background if background is not None else self.background_hist, float)
        if S.shape != B.shape:
            raise ValueError("signal и background должны иметь одинаковую длину")

        
        delta = delta if delta is not None else self.delta

        if delta is None:
            term = (S + B) * np.log((S + B) / B) - S
            return np.sqrt(2 * np.sum(term))

        else:
            d2 = delta**2
            
            t1 = (S + B) * np.log( ((S + B) * (1 + d2 * B)) /
                                   (B + d2 * B * (S + B)) )
            
            t2 = (1.0 / d2) * np.log(1 + d2 * S / (1 + d2 * B))
            return np.sqrt(2 * np.sum(t1 - t2))

    def _asymptotic_mu(self,
                       signal=None,
                       background=None,
                       delta=None,
                       cl=None):

        S_arr = np.array(signal     if signal     is not None else self.signal_hist, float)
        B_arr = np.array(background if background is not None else self.background_hist, float)
        if S_arr.shape != B_arr.shape:
            raise ValueError("signal и background должны иметь одинаковую длину")

        delta    = delta if delta is not None else self.delta
        alpha    = 1.0 - (cl if cl is not None else 0.95)
        Zcl  = norm.ppf(1.0 - alpha)   
        def Z_excl(mu):
            S = mu * S_arr
            B = B_arr

            if delta is None:
                
                term = S - B * np.log(1 + S/B)
                return np.sqrt(2 * np.sum(term))

            else:
                d2 = delta**2
                under = (B + S)**2 - (4*d2 * B**2 * S)/(d2*B + 1)
                x = np.sqrt(np.maximum(0.0, under))
                inner = ( S
                        - B * np.log((B + S + x)/(2*B))
                        - (1.0/d2) * np.log((B - S + x)/(2*B)) )
                outer = 2*inner - (B + S - x) * (1 + 1.0/(d2 * B))
                return np.sqrt(np.maximum(0.0, np.sum(outer)))

        def f(mu): 
            return Z_excl(mu) - Zcl

        mu_lo, mu_hi = 0.0, 1.0
        while f(mu_hi) < 0:
            mu_hi *= 2
            if mu_hi > 1e6:
                Print("Не удалось подобрать μ для поиска корня")
                return None
        mu_up = brentq(f, mu_lo, mu_hi, xtol=1e-6, maxiter=100)
        return mu_up



    def asymptotic_hypotest(self, data=None, signal_hist=None, background_hist=None, delta=None, cl=0.95,
     mode=None, prior=None
                ):
        signal = signal_hist if signal_hist else self.signal
        background = background_hist if background_hist else self.background
        data = data if data else self.data
        mode = mode if mode else self.mode
        if mode == 'significance':
            significance = self._asymptotic_significance(signal, background, delta)
            return significance
        elif mode == 'upper_limits':
            mu = self._asymptotic_mu(signal, background, delta, cl)
            return mu + 1
                
    def plot_hypo_test(self, path=None):
        import matplotlib.pyplot as plt

        if self.mode == 'significance':
            if not hasattr(self, '_q0_dist') or not hasattr(self, '_q0_obs'):
                raise RuntimeError("Monte Carlo test must be run before plotting (significance mode).")

            counts, bin_edges = np.histogram(self._q0_dist, bins=50, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)

            plt.figure(figsize=(8, 6))

            label_added = False
            for count, x, width in zip(counts, bin_centers, bin_widths):
                if x >= self._q0_obs:
                    if not label_added:
                        plt.bar(x, count, width=width, color='red', alpha=0.7, edgecolor='black',
                                label='Tail region (p-value)')
                        label_added = True
                    else:
                        plt.bar(x, count, width=width, color='red', alpha=0.7, edgecolor='black')
                else:
                    plt.bar(x, count, width=width, color='skyblue', alpha=0.7, edgecolor='black')

            plt.axvline(self._q0_obs, color='black', linestyle='--', linewidth=2, label=fr'$q_0^{{obs}} = {self._q0_obs:.2f}$')
            plt.xlabel(r'Test statistic $q_0$')
            plt.ylabel('Density')
            plt.title(r'Distribution of $q_0$ under $H_0$')
            p_val = getattr(self, '_p_val', None)
            significance = getattr(self, '_significance', None)
            if p_val is not None and significance is not None:
                plt.legend(title=fr'p = {p_val:.5f}, Z = {significance:.2f}σ', loc='upper right')
            else:
                plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            if path:
                plt.savefig(path)
            plt.show()

        elif self.mode == 'upper_limits':
            if not hasattr(self, '_p_values'):
                raise RuntimeError("Monte Carlo test must be run before plotting (upper_limits mode).")

            plt.figure(figsize=(8, 6))
            plt.plot(self.ul_grid[:len(self._p_values)], self._p_values, marker='o', label='p-value', color='blue')
            plt.axhline(1 - self.cl, color='red', linestyle='--', label=fr'CL = {self.cl:.2f}')
            plt.xlabel(r'Signal strength $\mu$')
            plt.ylabel('p-value')
            plt.title('p-value vs signal strength μ')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if path:
                plt.savefig(path)
            plt.show()

        else:
            raise ValueError("Unknown mode for plotting.")
    
    def plot_data(self, data=None, signal_hist=None, background_hist=None, delta=None, delta_s=None, path=None,
                    plot_signal=True, plot_background=True, plot_data=True
                        ):
        import matplotlib.pyplot as plt
        data = data if data else self.data
        signal = signal_hist if signal_hist else self.signal
        background = background_hist if background_hist else self.background
        if delta:
            errbg = background * delta
        if delta_s:
            errsig = signal * delta_s
        bins_range = [i for i in range(len(data))]
        plt.figure(figsize=(12,8))
        if plot_data:
            plt.scatter(bins_range, data, color = 'black', label = 'data')
        if plot_signal:
            plt.bar(bins_range, signal, color = 'blue', label = 'Signal events', alpha = 0.5, bottom=background if plot_background else None)
            if delta_s:
                plt.errorbar(bins_range, signal, errsig, fmt='none', color = 'black', label = f'Signal error (+- {delta_s*100}%)')
        if plot_background:
            plt.bar(bins_range, background, color = 'olive', label = 'Background events', alpha=0.5)
            if delta:
                plt.errorbar(bins_range, background, errbg, fmt='none', color = 'black', alpha=0.5, label = f'background error (+- {delta*100}%)')
        
        plt.xlabel('bins')
        plt.ylabel('n_events')
        plt.ylim(0, np.max(data)+np.std(data)*5)
        plt.legend()
        plt.grid(True)
        title = ''
        if plot_data:
            title += 'Distribution of data points'
        if plot_signal:
            title += ', Signal events'
        if plot_background:
            title += ', background events'
        plt.title(title)
        if path:
            plt.savefig(path)
        plt.show()
        
        return None