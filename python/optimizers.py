"""Drop-in optimizers: minimize objective(theta); count EVERY fitness call."""
import numpy as np

def _initialize(objective,lower,upper,initial,opts):
    lo=np.asarray(lower); hi=np.asarray(upper); initial=np.asarray(initial)
    N=opts['population']; budget=opts['max_evals']; rng=np.random.default_rng(opts['seed'])
    if N<2 or budget<N or not np.all(hi>lo): raise ValueError('Invalid search configuration')
    pop=lo+rng.random((N,len(lo)))*(hi-lo); pop[0]=np.clip(initial,lo,hi)
    values=np.array([objective(x.copy()) for x in pop])
    if not np.isfinite(values).all(): raise ValueError('Nonfinite objective')
    i=values.argmin(); best=float(values[i]); xbest=pop[i].copy()
    history=np.minimum.accumulate(values).tolist()
    return rng,pop,values,best,xbest,history

def _result(best,xbest,history):
    return dict(best_position=xbest,best_cost=best,history=np.asarray(history),evals=len(history))

def optimizer_ga(objective,lower,upper,initial,opts):
    rng,pop,costs,best,xbest,history=_initialize(objective,lower,upper,initial,opts)
    N,D=pop.shape; budget=opts['max_evals']; span=upper-lower
    while len(history)<budget:
        nxt=pop.copy(); ncost=costs.copy(); nxt[0]=xbest; ncost[0]=best
        for i in range(1,N):
            if len(history)>=budget: break
            a=rng.integers(N,size=2); b=rng.integers(N,size=2)
            p1=pop[a[costs[a].argmin()]];p2=pop[b[costs[b].argmin()]]
            alpha=rng.random(D); child=alpha*p1+(1-alpha)*p2
            mask=rng.random(D)<max(1/D,.05);step=.1*(1-.8*len(history)/budget)
            child=np.clip(child+mask*rng.normal(size=D)*span*step,lower,upper)
            value=float(objective(child.copy()))
            if not np.isfinite(value): raise ValueError('Nonfinite GA objective')
            nxt[i]=child;ncost[i]=value
            if value<best: best=value;xbest=child.copy()
            history.append(best)
        pop,costs=nxt,ncost
    return _result(best,xbest,history)

def optimizer_pso(objective,lower,upper,initial,opts):
    rng,pop,costs,best,xbest,history=_initialize(objective,lower,upper,initial,opts)
    N,D=pop.shape; budget=opts['max_evals']; span=upper-lower
    pbest=pop.copy(); velocity=np.zeros_like(pop)
    while len(history)<budget:
        for i in range(N):
            if len(history)>=budget: break
            velocity[i]=.7*velocity[i]+1.5*rng.random(D)*(pbest[i]-pop[i])+1.5*rng.random(D)*(xbest-pop[i])
            velocity[i]=np.clip(velocity[i],-.2*span,.2*span)
            proposal=pop[i]+velocity[i]; hit=(proposal<lower)|(proposal>upper)
            pop[i]=np.clip(proposal,lower,upper);velocity[i,hit]=0
            value=float(objective(pop[i].copy()))
            if not np.isfinite(value): raise ValueError('Nonfinite PSO objective')
            if value<costs[i]:costs[i]=value;pbest[i]=pop[i].copy()
            if value<best:best=value;xbest=pop[i].copy()
            history.append(best)
    return _result(best,xbest,history)

def optimizer_xx_template(objective,lower,upper,initial,opts):
    """Return best_position, best_cost, history (per evaluation), evals.

    opts keys: population, max_evals, seed. Optimize only the passed objective.
    This intentionally raises until the student implements their own heuristic.
    """
    raise NotImplementedError('Implement your selected heuristic here')


def optimizer_gso(objective, lower, upper, initial, opts):
    """Glider Snake Optimizer (GSO) — drop-in implementation for Neuro-Fuzzy-Learning-Lab."""
    rng, pop, costs, best, xbest, history = _initialize(objective, lower, upper, initial, opts)
    N, D = pop.shape
    budget = opts['max_evals']
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    while len(history) < budget:
        progress = len(history) / float(budget)
        glide_factor = 2.0 * (1.0 - progress)

        order = np.argsort(costs)
        pop = pop[order]
        costs = costs[order]

        for i in range(N):
            if len(history) >= budget:
                break

            prev_snake = xbest if i == 0 else pop[i - 1]

            if rng.random() < 0.5:
                r1 = rng.random(D)
                r2 = rng.random(D)
                wave = np.sin(2.0 * np.pi * r1)
                drift = np.cos(2.0 * np.pi * r2)
                step = (
                    glide_factor * wave * (xbest - pop[i])
                    + glide_factor * drift * (prev_snake - pop[i])
                )
                proposal = pop[i] + step
            else:
                r3 = rng.random(D)
                r4 = rng.random(D)
                proposal = (
                    xbest
                    + glide_factor * r3 * (xbest - pop[i])
                    + (0.5 * glide_factor) * r4 * (prev_snake - pop[i])
                )

            proposal = np.clip(proposal, lo, hi)

            value = float(objective(proposal.copy()))
            if not np.isfinite(value):
                raise ValueError('Nonfinite GSO objective')

            if value < costs[i]:
                pop[i] = proposal
                costs[i] = value
                if value < best:
                    best = value
                    xbest = proposal.copy()

            history.append(best)

    return _result(best, xbest, history)

    def optimizer_gsodbo(objective, lower, upper, initial, opts):
    """
    Hibrit Glider Snake + Dung Beetle Optimizer (GSODBO)
    ---------------------------------------------------
    Ana iskelet : Glider Snake Optimizer (GSO)
    DBO katkısı : (1) En kötü çözümden kaçış (ball-rolling itme terimi),
                  (2) En iyi çözüm etrafında dinamik daralan sömürü sınırı (breeding),
                  (3) Zayıf bireyler için hırsız (thief) yerel arama adımı.
    Bütçe       : Ekstra amaç fonksiyonu çağrısı yapmadan max_evals (320 FEs) ile çalışır.
    """
    rng, pop, costs, best, xbest, history = _initialize(objective, lower, upper, initial, opts)
    N, D = pop.shape
    budget = opts['max_evals']
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    span = hi - lo

    # Popülasyonun en zayıf %20'lik dilimi için eşik indeks (DBO hırsız grubu)
    thief_start = max(1, int(0.8 * N))

    while len(history) < budget:
        progress = len(history) / float(budget)
        R = 1.0 - progress              # DBO dinamik daralma katsayısı (1 -> 0)
        glide_factor = 2.0 * R          # GSO süzülme katsayısı (2 -> 0)

        # Bireyleri mevcut başarısına göre sırala: pop[0] en iyi, pop[-1] en kötü
        order = np.argsort(costs)
        pop = pop[order]
        costs = costs[order]
        xworst = pop[-1]

        # DBO dinamik daralan bölge sınırları (negatif parametrelere uyumlu simetrik yapı)
        half_box = 0.5 * R * span
        lb_star = np.maximum(xbest - half_box, lo)
        ub_star = np.minimum(xbest + half_box, hi)

        for i in range(N):
            if len(history) >= budget:
                break

            prev_snake = xbest if i == 0 else pop[i - 1]

            if i >= thief_start:
                # DBO Fazı: Zayıf yılanlar için Hırsız (Thief) yerel araması
                g = rng.normal(size=D)
                S = 0.5 * R
                proposal = xbest + S * g * (
                    np.abs(pop[i] - xbest) + np.abs(pop[i] - prev_snake)
                )
            elif rng.random() < 0.5:
                # Hibrit Keşif: GSO sinüzoidal süzülme + DBO en kötüden uzaklaşma
                r1 = rng.random(D)
                r2 = rng.random(D)
                r_rep = rng.random(D)
                gso_step = (
                    glide_factor * np.sin(2.0 * np.pi * r1) * (xbest - pop[i])
                    + glide_factor * np.cos(2.0 * np.pi * r2) * (prev_snake - pop[i])
                )
                dbo_avoid = 0.15 * glide_factor * r_rep * (pop[i] - xworst)
                proposal = pop[i] + gso_step + dbo_avoid
            else:
                # Hibrit Sömürü: DBO dinamik daralan bölge (breeding) + GSO kılavuzluğu
                b1 = rng.random(D)
                b2 = rng.random(D)
                proposal = xbest + b1 * (pop[i] - lb_star) + b2 * (pop[i] - ub_star)
                proposal = np.clip(proposal, lb_star, ub_star)

            # Genel parametre sınırları içine çekme
            proposal = np.clip(proposal, lo, hi)

            # Amaç fonksiyonu değerlendirmesi (her çağrı bütçeden sayılır)
            value = float(objective(proposal.copy()))
            if not np.isfinite(value):
                raise ValueError('Nonfinite GSODBO objective')

            # Açgözlü seçim (greedy selection)
            if value < costs[i]:
                pop[i] = proposal
                costs[i] = value
                if value < best:
                    best = value
                    xbest = proposal.copy()

            history.append(best)

    return _result(best, xbest, history)
    
    raise NotImplementedError('Implement your selected heuristic here')
