# Spiegazione del Kernel Trick - Data Mining e Machine Learning

Questo documento riassume i concetti illustrati dallo script `DISD - Kernel Trick.R` (e dal corrispondente file `.html`), focalizzandosi sul funzionamento intuitivo del "Trucco del Kernel" usato nelle **Support Vector Machines (SVM)**.

## 1. Il Problema: Dati non linearmente separabili
Lo script genera un piccolo dataset giocattolo (in 2D, con coordinate spaziali `x1` e `x2`) e una variabile di classe `y` (valori 1 e -1, visualizzati rispettivamente in blu e rosso).
Visualizzando il classico grafico a dispersione bidimensionale, i punti blu si trovano tutti raggruppati vicino all'origine (0,0), mentre i punti rossi formano un "anello" attorno a essi.
In questa situazione, è **impossibile tracciare una singola linea retta** che divida perfettamente i due gruppi. Si dice, infatti, che i dati *non sono linearmente separabili* in due dimensioni.

## 2. La Soluzione: Aumentare le Dimensioni (Feature-vector mapping)
Visto che un taglio lineare in 2D è impossibile, il trucco consiste nel sollevare e proiettare i dati in uno spazio con una dimensione extra (3D).
Il codice crea fisicamente questa nuova terza dimensione calcolandola:
```r
feat = x1^2 + x2^2
```
In termini puramente geometrici, `x1^2 + x2^2` calcola esattamente il quadrato della distanza di ogni punto dal centro del grafico.

## 3. La visualizzazione in 3D (Plotly)
Il professore utilizza il pacchetto `plotly` per generare un grafico 3D navigabile usando le coordinate (`x1`, `x2`, `feat`). In questo nuovo spazio, la morfologia dei dati cambia radicalmente:
* I **punti blu**, trovandosi originariamente vicino al centro, ottengono un valore di `feat` bassissimo e restano quindi ancorati al "fondo" del grafico.
* I **punti rossi**, trovandosi più distanti dall'origine, ottengono un valore di `feat` grandissimo. Vengono letteralmente "sparati" verso l'alto lungo l'asse Z, formando le pareti di un paraboloide (una sorta di scodella).

**Risultato Magistrale**: Nello spazio 3D i rossi e i blu sono perfettamente separati sul piano verticale! Ora basterà far passare un foglio di carta orizzontale e piatto (un *iperpiano*) a metà altezza per separare perfettamente e senza errori le due classi.

## In Sintesi: Il Kernel Trick
Il Kernel Trick è l'essenza di questo processo. Quando un dataset è aggrovigliato e non può essere classificato da un confine dritto, una funzione matematica (chiamata **Kernel**) si occupa di proiettare i dati in uno spazio a dimensioni superiori (in alcuni casi, anche a infinite dimensioni). In questo nuovo spazio espanso, i dati si "srotolano" e diventano facilmente separabili tramite un piano lineare piatto. 
Si chiama "Trick" (trucco) perché le equazioni delle SVM riescono a simulare questa proiezione e calcolare le separazioni ottimali *senza* dover materialmente aggiungere e processare enormi quantità di nuove colonne di calcolo nel dataset, risultando così computazionalmente elegantissime e fulminee.
