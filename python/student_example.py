"""Working integration example. Replace optimizer_pso with your optimizer_xx."""
from course_core import (Config, examples, prepare_data, init_fnn, train_meta_fnn,
                         predict, classification_metrics, rmse)
from optimizers import optimizer_pso

def run_student(optimizer=optimizer_pso, cfg=None, datasets=None):
    cfg=cfg or Config(); rows=[]
    for ex in examples() if datasets is None else datasets:
        data=prepare_data(**{k:v for k,v in ex.items() if k!='name'})
        for seed in cfg.seeds:
            initial=init_fnn(data['Xtrain'],cfg.rules,1,seed)
            model,search=train_meta_fnn(initial,data['Xtrain'],data['ytrain'],
                data['Xval'],data['yval'],cfg,seed,optimizer)
            pred=predict(model,data['Xtest'])*data['ystd']+data['ymean']
            metrics=classification_metrics(data['ytest'],pred,data['classes']) if ex['task']=='classification' \
                else dict(rmse=rmse(pred.ravel(),data['ytest']))
            print(ex['name'],seed,metrics)
            rows.append(dict(example=ex['name'],seed=seed,model=model,search=search,metrics=metrics))
    return rows

def optimizer_gso(objective, lower, upper, initial, opts):
    """Glider Snake Optimizer (GSO) — drop-in implementation for Neuro-Fuzzy-Learning-Lab."""
    rng, pop, costs, best, xbest, history = _initialize(objective, lower, upper, initial, opts)
    N, D = pop.shape
    budget = opts['max_evals']
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    while len(history) < budget:
        # Süreç ilerledikçe (0 -> 1) keşiften (süzülme) sömürüye (iniş) geçiş
        progress = len(history) / float(budget)
        glide_factor = 2.0 * (1.0 - progress)  # 2'den 0'a doğrusal azalma

        # Bireyleri mevcut başarısına göre sırala (en iyi yılan indeks 0)
        order = np.argsort(costs)
        pop = pop[order]
        costs = costs[order]

        for i in range(N):
            if len(history) >= budget:
                break

            # Çift kılavuz: global en iyi (xbest) ve bir üst/önceki yılan (prev_snake)
            prev_snake = xbest if i == 0 else pop[i - 1]

            if rng.random() < 0.5:
                # Faz 1: Süzülme / Keşif (sinüzoidal yanal dalgalanma + çift kılavuz)
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
                # Faz 2: Hedefli İniş / Sömürü (en iyi yılan etrafında daralan arama)
                r3 = rng.random(D)
                r4 = rng.random(D)
                proposal = (
                    xbest
                    + glide_factor * r3 * (xbest - pop[i])
                    + (0.5 * glide_factor) * r4 * (prev_snake - pop[i])
                )

            # Parametre sınırları içine çekme
            proposal = np.clip(proposal, lo, hi)

            # Amaç fonksiyonu değerlendirmesi (her çağrı bütçeden düşer)
            value = float(objective(proposal.copy()))
            if not np.isfinite(value):
                raise ValueError('Nonfinite GSO objective')

            # Açgözlü seçim (greedy selection)
            if value < costs[i]:
                pop[i] = proposal
                costs[i] = value
                if value < best:
                    best = value
                    xbest = proposal.copy()

            history.append(best)

    return _result(best, xbest, history)

if __name__=='__main__':run_student()
