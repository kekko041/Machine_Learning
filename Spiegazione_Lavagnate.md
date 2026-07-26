# Spiegazione Lavagnate - Data Mining e Machine Learning

Questo documento trascrive e spiega il contenuto delle lavagnate scritte a mano dal Prof. Brutti durante le lezioni. Sono note teoriche fondamentali che accompagnano il codice R dei vari day.

---

## Day 2 – Lavagnata 1 (2 foto)

### Foto 1 – Il Setup del Problema di Apprendimento Supervisionato
La prima lavagnata introduce il formalismo matematico generale dell'apprendimento supervisionato:

- **y = var. risposta** e **{x₁, ..., xₚ} = var. esplicative**: definizione delle variabili del problema.
- **𝒟ₙ = {(y₁,x₁), ..., (yₙ,xₙ)}**: il dataset di apprendimento con n=215 unità (i pezzi di carne) e p=100 variabili esplicative (le frequenze spettrometriche).
- **y num → Regressione** / **y categoriale → Classificazione**: distinzione fondamentale tra i due tipi di apprendimento supervisionato.
- Il modello generale è una funzione parametrizzata: **fθ(x₁,...,xₚ) = θ₀ + θ₁φ₁(x) + θ₂φ₂(x) + ... + θₖφₖ(x) = θᵀΦ(x)**, dove **Φ(x)** è il vettore delle *feature* (trasformazioni delle variabili originali, che possono essere tante - **k >> p**).
- L'obiettivo è trovare il vettore di parametri ottimale **θ̂** tramite la minimizzazione della somma degli errori al quadrato sul Training set: **θ̂ = argmin Σᵢ (yᵢ - fθ(xᵢ))²**.

### Foto 2 – K-Fold Cross-Validazione
La seconda foto della lavagnata del Day 2 illustra graficamente il meccanismo della K-Fold Cross-Validation (**K-fold CV**):

- Il dataset di 215 osservazioni viene diviso in **Training (70%) e Test (30%)**.
- Il Training set viene ulteriormente suddiviso in **K=3 blocchi (fold)**.
- A ogni iterazione, un fold diverso viene usato come **"Validation set"** (indicato con `Val`) e i restanti K-1 fold vengono usati per addestrare il modello (`Tr`).
- Per ogni valore del parametro di tuning (es. il cut-off di correlazione), si ottiene un errore di validazione: **e₁, e₂, e₃**.
- Il CV-score finale è la **media di questi errori**: `ave(e₁, e₂, e₃)`.
- La nota "per una griglia di valori del par. di tuning, ripeto" indica che questa procedura si ripete per ogni valore della griglia, e si sceglie il parametro con il CV-score più basso.

---

## Day 4 – Lavagnate (3 pagine, 6 foto)

Tutte le lavagnate del Day 4 trattano i **Metodi Penalizzati** (Regularization), un argomento centrale che risponde al problema dell'overfitting e della multicollinearità visto nei Day 1 e 2.

### Pagina 1 – Motivazione e le Norme

**Foto 1 – Il Problema e la Motivazione**
- **Ricordiamo**: adattando il modello completo con tutte le variabili, data l'alta correlazione fra di esse, i valori dei coefficienti di regressione erano fuori controllo (es. `-17t, +81e3, ...`).
- La soluzione: bisogna aggiungere una misura di "grandezza dei coefficienti θ" (chiamata **NORMA(θ)**) al problema di ottimizzazione, per evitare che i coefficienti "esplodano".

**Foto 2 – Le Norme matematiche**
La nota introduce le diverse norme usate come penalità:
- **Norma Euclidea (L₂)**: `‖θ‖₂ = √(Σⱼθₖ²)` → penalizza la lunghezza euclidea del vettore dei pesi. Rappresentata come **cerchio/sfera** nello spazio dei parametri.
- **Norma L₁**: `‖θ‖₁ = Σⱼ|θⱼ|` → penalizza la somma dei valori assoluti. Rappresentata come un **rombo/diamante** nello spazio dei parametri.
- **Norma Lₚ generale**: `‖θ‖ₚ = (Σⱼ|θⱼ|ᵖ)^(1/p)`.
- **Norma L∞**: `‖θ‖∞ = max|θ|`.
- **Norma L₀**: `‖θ‖₀ = #{θ ≠ 0}` (conta i coefficienti non nulli → Subset Selection). I disegni mostrano le "palle" geometriche di queste norme: la L₂ è un cerchio liscio, la L₁ è un quadrato ruotato (rombo), la L∞ è un quadrato.

### Pagina 2 – Forma Generale e Ridge vs Lasso

**Foto 3 – La Forma del Problema Penalizzato**
L'idea unificante: si vuole **vincolare la ricerca** di θ. Il problema diventa:
> `θ̂ = argmin Σᵢ(yᵢ - fθ(xᵢ))² + λ · ‖θ‖`

dove **λ > 0** è il parametro di tuning scelto via cross-validazione. La lavagnata mostra la connessione con le norme:
- **L₂ → Ridge (Weight Decay)**
- **L₁ → Lasso (Subset Selection)**
- **L₀ → Subset Selection** (NP-hard da ottimizzare, difficilissimo computazionalmente)

**Foto 4 – L₂ vs L₁: Denso vs Sparso**
La differenza cruciale tra i due approcci:
- **Ridge (L₂)**: La soluzione è **densa**. Tutti i coefficienti vengono ridotti verso lo zero, ma nessuno viene portato **esattamente** a zero. `θ₁ = [1/√k, 1/√k, ..., 1/√k]` → distribuisce equamente l'importanza.
- **Lasso (L₁)**: La soluzione è **sparsa**. Molti coefficienti vengono portati esattamente a zero, realizzando una vera **selezione automatica delle variabili**. `θ₂ = [1, 0, 0, ..., 0]` → seleziona le variabili più importanti.
- Il disegno geometrico a destra mostra intuitivamente perché: la "palla" L₂ (cerchio liscio) tocca l'ellisse della funzione obiettivo in un punto interno (non sugli assi → coeff ≠ 0), mentre la "palla" L₁ (rombo) tocca l'ellisse su un **vertice** (sugli assi → il coefficiente corrispondente diventa esattamente 0).

### Pagina 3 – Le Soluzioni Analitiche

**Foto 5 – Ridge Regression (L₂)**
Il problema Ridge in forma matriciale:
> `θ̂(λ) = argmin ‖y - Xθ‖₂² + λ·‖θ‖₂²`

La soluzione analitica si ottiene derivando e ponendo uguale a zero:
> `θ̂ᴿ(λ) = (XᵀX + λI)⁻¹Xᵀy`

La nota chiave: al crescere di λ, i coefficienti **devono andare a zero** (Ridge path). Il termine `λI` stabilizza la matrice `XᵀX` (che era instabile/singolare a causa della multicollinearità) rendendo sempre possibile l'inversione.

**Foto 6 – Lasso Regression (L₁)**
Il problema Lasso in forma matriciale:
> `θ̂(λ) = argmin ‖y - Xθ‖₂² + λ·‖θ‖₁`

Non esiste una soluzione analitica in forma chiusa (la derivata della norma L₁ non è definita in θ=0). La soluzione si chiama **Lasso path** e, al crescere di λ, alcuni coefficienti vengono portati **esattamente a zero** (selezionando le variabili), come annotato "alcuni θ saranno esattamente zero".

---

## Day 5 – Lavagnate (3 pagine, 6 foto)

### Pagina 1 – Il Framework della Classificazione

**Foto 1 – Setup del Problema**
- **Classificazione = prevedere una variabile categoriale**, trattata come numero per comodità (es. y = "low"/"high" oppure 0/+1 o -1/+1).
- Il classificatore è `hθ(x)`. La domanda è: che metrica usare per confrontare `y` con `hθ(x)`?
- **Perdita 0/1**: `L(θ) = (1/n)Σ 𝟙(yᵢ ≠ hθ(xᵢ))` → conta la proporzione di errori di classificazione.
- `θ̂ = argmin L(θ)` → addestramento = minimizzare la perdita sul Training set.

**Foto 2 – Osservazioni importanti sulla Perdita 0/1**
Ci sono 3 note cruciali sulla perdita 0/1:
1. In pratica viene **sostituita** con surrogati più semplici da ottimizzare matematicamente (es. **Hinge loss** per SVM, Cross-Entropy per Reti Neurali).
2. La perdita 0/1 valutata sul training è ottimistica → non distingue la "natura" degli errori (Falsi Positivi vs Falsi Negativi).
3. Meglio considerare **funzioni di perdita basate su Sensitivity/Specificity/AUC/ROC/F1** per valutare in modo più completo il classificatore.

### Pagina 2 – Naive Bayes e Alberi di Classificazione

**Foto 3 – Naive Bayes (Gaussiano)**
Introduce il classificatore **Naive Bayes Gaussiano**, basato sul Teorema di Bayes:
- `Pr(Y|X)` ∝ `Pr(X|Y) · Pr(Y)` (probabilità congiunta)
- Si assume che `Pr(X|Y=low) ~ N(μ_low, Σ_low)` e `Pr(X|Y=high) ~ N(μ_high, Σ_high)`.
- Il grafico a destra mostra un dataset a cerchi concentrici (come quello del Classifiers Zoo), dimostrando le regioni di decisione di Naive Bayes.

**Foto 4 – Alberi di Decisione (CART)**
- Il disegno mostra la struttura ad albero binario: ogni nodo fa una domanda del tipo `x₁ ≤ 5` → se sì a sinistra, se no a destra.
- Il confine di decisione nel piano (x₁, x₂) risultante sono **rettangoli** (gradini), come visto nel Classifiers Zoo.
- **Ensemble di alberi → molto più stabile!** → **RF (Random Forest)** e **GB (Gradient Boosting)**.

### Pagina 3 – Bias vs Varianza: RF vs Gradient Boosting

**Foto 5 e 6 – "Back in the days": Distorsione vs Variabilità**
La lavagnata più teorica del corso. Torna sul trade-off Bias-Varianza applicandolo agli Ensemble:

- La classica curva a **U rovesciata** dell'errore di previsione (asse Y) in funzione della complessità del modello (asse X = `depth` dell'albero):
  - A sinistra (alberi semplici): alto **Bias**, bassa Varianza → Underfitting.
  - A destra (alberi profondi): basso Bias, alta **Varianza** → Overfitting.
  - Il punto ottimale è in mezzo.

- **Random Forest (RF)**:
  - Si parte da alberi **overfittanti** (alta Varianza, basso Bias).
  - L'Ensembling (media/voto) deve **controllare la Varianza**.
  - Trucchi: ① Media di B previsioni indipendenti (o Bootstrap → OOB) → variabilità ridotta. ② Selezione random delle variabili di split.
  
- **Gradient Boosting (GB)**:
  - Si parte da alberi **underfittanti** (alta Distorsione, bassa Varianza).
  - L'Ensembling deve **controllare il Bias**.
  - Idea: "scaviamo nei residui" ad ogni iterazione (si fitta il modello successivo sull'errore residuo del precedente).
  - Con M "riscampionamenti" successivi `D⁽¹⁾, ..., D⁽ᴹ⁾`, sempre bassa distorsione ma variabilità ridotta.
