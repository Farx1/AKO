# Adaptive Kernel Blocks for Hardware-Specific LLM Runtime Optimization

**Version:** 0.1  
**Format:** équivalent mini-papier de recherche  
**Objectif:** formaliser mathématiquement pourquoi une bibliothèque de kernels adaptatifs peut accélérer certaines opérations de modèles IA, puis proposer un protocole de validation expérimental.

---

## Résumé

Les modèles de type Transformer/LLM sont dominés par un petit nombre d’opérations répétées : multiplications matricielles, attention, normalisation, RoPE, quantization/dequantization et gestion du KV cache. Les runtimes génériques comme PyTorch, TensorRT-LLM, vLLM ou des backends CUDA/Triton très optimisés couvrent déjà de nombreux cas, mais ils doivent rester relativement génériques.

Ce document propose une approche complémentaire : créer une bibliothèque de **blocs bas niveau adaptatifs**, spécialisés par GPU, shape d’entrée et précision numérique. L’objectif n’est pas de réécrire toute une stack LLM en assembleur, mais de remplacer certaines opérations critiques par des kernels spécialisés, sélectionnés automatiquement par benchmark ou autotuning.

Nous montrons que cette approche est théoriquement défendable lorsque l’opération est **memory-bound** ou lorsque plusieurs opérations peuvent être **fusionnées**, car le gain provient surtout d’une réduction du trafic mémoire entre la mémoire globale GPU, souvent HBM, et les mémoires rapides du GPU.

---

## 1. Intuition générale

L’idée de départ peut être formulée ainsi :

> Au lieu d’avoir une seule implémentation générique d’une opération LLM, on maintient plusieurs implémentations spécialisées, puis on choisit automatiquement la meilleure selon le GPU, la shape, le dtype et le mode d’inférence.

Le système cible ressemble à ceci :

```text
Opération logique
    ↓
Candidats kernels
    ↓
Autotuning / microbenchmark
    ↓
Validation numérique
    ↓
Sélection du meilleur kernel
    ↓
Cache de décision
    ↓
Exécution runtime
```

Ce n’est donc pas un projet “tout assembleur”. C’est plutôt un projet de type :

> **adaptive kernel backend for LLM inference and AI model optimization**

ou, en français :

> **backend adaptatif de kernels spécialisés pour modèles IA**.

---

## 2. Contributions possibles du projet

Les contributions techniques pourraient être :

1. Une bibliothèque de blocs bas niveau spécialisés : RMSNorm, RoPE, matmul quantifié, KV cache, attention decode.
2. Un système de sélection automatique du meilleur kernel selon le GPU, la shape et le dtype.
3. Un protocole de validation numérique pour éviter les régressions.
4. Une logique d’autotuning et de cache des meilleurs kernels.
5. Une intégration possible dans une pipeline MLOps.
6. Un rapport benchmark automatique comparant baseline et version optimisée.

La contribution scientifique n’est pas seulement d’écrire un kernel rapide. Elle est dans la combinaison :

```text
spécialisation hardware
+ autotuning
+ validation numérique
+ fallback fiable
+ benchmark reproductible
+ intégration MLOps
```

---

## 3. Modèle de performance

On considère une opération \( o \), une shape d’entrée \( s \), un GPU \( h \), et un kernel candidat \( k \).

On définit :

\[
F_o(s)
\]

comme le nombre d’opérations arithmétiques nécessaires.

On définit :

\[
Q_k(h, s)
\]

comme le volume de données transférées entre la mémoire globale GPU et les mémoires rapides/cache/registers/shared memory.

On définit :

\[
P_h
\]

comme le débit arithmétique maximal ou effectif du GPU, en FLOP/s.

On définit :

\[
\beta_h
\]

comme la bande passante mémoire effective du GPU, en bytes/s.

Le temps d’exécution d’un kernel est borné par un modèle de type roofline :

\[
T_k(h, s)
\geq
\max\left(
\frac{F_o(s)}{P_h},
\frac{Q_k(h,s)}{\beta_h}
\right)
+ \lambda_k
\]

où \( \lambda_k \) représente l’overhead de lancement du kernel et les coûts fixes.

---

## 4. Cas compute-bound et memory-bound

### 4.1 Cas compute-bound

Si :

\[
\frac{F_o(s)}{P_h}
\gg
\frac{Q_k(h,s)}{\beta_h}
\]

alors l’opération est **compute-bound**.

Dans ce cas, réduire le trafic mémoire ne donne pas forcément un gros gain. Les bons leviers sont plutôt :

- réduire le nombre d’opérations arithmétiques ;
- utiliser une précision plus efficace, par exemple FP8, INT8 ou INT4 ;
- exploiter les Tensor Cores ;
- améliorer le tiling et l’occupation GPU.

### 4.2 Cas memory-bound

Si :

\[
\frac{Q_k(h,s)}{\beta_h}
\gg
\frac{F_o(s)}{P_h}
\]

alors l’opération est **memory-bound**.

Dans ce cas, si un kernel optimisé \( k^\* \) réduit le trafic mémoire :

\[
Q_{k^\*}(h,s) < Q_{base}(h,s)
\]

alors le speedup théorique est approximativement :

\[
S(h,s)
=
\frac{T_{base}}{T_{k^\*}}
\approx
\frac{Q_{base}(h,s)}{Q_{k^\*}(h,s)}
\]

C’est le cœur mathématique du projet.

---

## 5. Théorème principal : sélection adaptative

Soit \( \mathcal{K}_o \) l’ensemble des kernels disponibles pour une opération \( o \).

On inclut toujours le fallback générique :

\[
k_{base} \in \mathcal{K}_o
\]

Le sélecteur adaptatif choisit :

\[
k^\*(h,s)
=
\arg\min_{k \in \mathcal{K}_o} T_k(h,s)
\]

Après autotuning et mise en cache de la décision, le runtime adaptatif exécute :

\[
T_{AKO}(h,s)
=
T_{k^\*}(h,s) + \delta
\]

où \( \delta \) est le coût de dispatch runtime.

Comme \( k_{base} \in \mathcal{K}_o \), on a :

\[
T_{k^\*}(h,s)
\leq
T_{base}(h,s)
\]

Donc :

\[
T_{AKO}(h,s)
\leq
T_{base}(h,s) + \delta
\]

Si \( \delta \approx 0 \), alors :

\[
T_{AKO}(h,s)
\leq
T_{base}(h,s)
\]

En pratique, le système est donc **non-pire que la baseline à overhead près**, à condition d’inclure un fallback fiable et de ne pas choisir un kernel numériquement incorrect.

---

## 6. Coût d’autotuning amorti

L’autotuning a un coût initial :

\[
C_{tune}(h,s)
\]

Mais ce coût est payé une seule fois pour un couple :

\[
(h, s, dtype, operation)
\]

Si le kernel sélectionné est réutilisé \( R \) fois, le temps moyen devient :

\[
\bar{T}_{AKO}
=
T_{k^\*}(h,s)
+
\delta
+
\frac{C_{tune}(h,s)}{R}
\]

Quand :

\[
R \to \infty
\]

alors :

\[
\bar{T}_{AKO}
\to
T_{k^\*}(h,s) + \delta
\]

Conclusion : le coût d’autotuning est acceptable pour :

- de l’inférence répétée ;
- du batch serving ;
- des modèles déployés longtemps ;
- des pipelines d’inférence réutilisées ;
- du benchmarking industriel.

Il est moins pertinent pour une seule inférence isolée.

---

## 7. Opération 1 — RMSNorm

On considère une matrice :

\[
X \in \mathbb{R}^{N \times D}
\]

où :

\[
N = B \times L
\]

avec :

- \( B \) : batch size ;
- \( L \) : sequence length ;
- \( D \) : hidden dimension.

RMSNorm est défini par :

\[
Y_{i,j}
=
\frac{X_{i,j}}{\sqrt{\frac{1}{D}\sum_{t=1}^{D}X_{i,t}^2 + \epsilon}}
\gamma_j
\]

### 7.1 Baseline non fusionnée

Une implémentation naïve fait typiquement :

1. calcul de \( X^2 \) ;
2. réduction par ligne ;
3. calcul du facteur inverse ;
4. multiplication finale par \( \gamma \).

En trafic mémoire simplifié, avec \( b \) bytes par élément :

```text
read X          → ND b
write X²        → ND b
read X²         → ND b
read X again    → ND b
read gamma      → ND b
write Y         → ND b
```

Donc :

\[
Q_{base}
\approx
6ND b
\]

### 7.2 Kernel fusionné

Un kernel optimisé charge une ligne, calcule la somme des carrés, calcule le facteur RMS, puis écrit directement la sortie.

Si la ligne tient dans les mémoires rapides/registers/shared memory :

```text
read X      → ND b
read gamma  → ND b
write Y     → ND b
```

Donc :

\[
Q_{fused}
\approx
3ND b
\]

Le speedup memory-bound maximal est :

\[
S_{RMSNorm}
\approx
\frac{Q_{base}}{Q_{fused}}
=
\frac{6ND b}{3ND b}
=
2
\]

Si la ligne doit être relue parce qu’elle ne tient pas entièrement dans les mémoires rapides :

\[
Q_{fused}
\approx
4ND b
\]

Alors :

\[
S_{RMSNorm}
\approx
\frac{6}{4}
=
1.5
\]

Conclusion théorique :

\[
1.5 \leq S_{RMSNorm} \leq 2
\]

sous hypothèse memory-bound.

---

## 8. Opération 2 — RoPE fusionné

RoPE applique une rotation aux dimensions paires/impaires de \( Q \) et \( K \).

Pour une paire :

\[
x'_{2i}
=
x_{2i}\cos\theta_i
-
x_{2i+1}\sin\theta_i
\]

\[
x'_{2i+1}
=
x_{2i}\sin\theta_i
+
x_{2i+1}\cos\theta_i
\]

### 8.1 Baseline séparée

Si RoPE est appliqué comme opération indépendante, on lit \( Q \) et \( K \), on lit les tables \( \cos, \sin \), puis on écrit \( Q' \) et \( K' \).

Ensuite, l’attention relit \( Q' \) et \( K' \).

Soit :

\[
X = 2BLHd b
\]

le volume mémoire de \( Q \) et \( K \), avec :

- \( B \) : batch ;
- \( L \) : longueur de séquence ;
- \( H \) : nombre de têtes ;
- \( d \) : dimension par tête.

La baseline effectue approximativement :

```text
RoPE read Q,K        → X
RoPE write Q',K'     → X
Attention read Q',K' → X
```

Donc :

\[
Q_{base}
\approx
3X + Q_{\cos,\sin}
\]

### 8.2 RoPE fusionné

Si RoPE est fusionné directement dans l’attention ou dans l’écriture du KV cache :

```text
read Q,K once
apply rotation in registers
consume immediately
```

Alors :

\[
Q_{fused}
\approx
X + Q_{\cos,\sin}
\]

Si \( Q_{\cos,\sin} \) est petit, préchargé, ou calculé efficacement, on obtient :

\[
S_{RoPE}
\approx
\frac{3X}{X}
=3
\]

Conclusion :

\[
S_{RoPE} \leq 3
\]

sur la partie RoPE + consommation immédiate.

Ce n’est pas forcément un speedup \( 3\times \) sur tout le modèle. C’est un speedup sur le sous-graphe concerné.

---

## 9. Opération 3 — Matmul quantifié INT4

On considère une couche linéaire :

\[
Y = XW
\]

avec :

\[
X \in \mathbb{R}^{B \times K}
\]

\[
W \in \mathbb{R}^{K \times M}
\]

En decode LLM avec petit batch, cette opération est souvent dominée par la lecture des poids.

### 9.1 Poids FP16

Les poids FP16 coûtent :

\[
Q_{W,fp16}
=
2KM
\]

bytes.

### 9.2 Poids INT4

Les poids INT4 coûtent :

\[
Q_{W,int4}
=
\frac{1}{2}KM
\]

bytes.

Il faut ajouter les scales de quantization. Si on note \( \eta \) l’overhead relatif des scales :

\[
Q_{W,int4,total}
=
\frac{1}{2}KM(1+\eta)
\]

Le speedup memory-bound théorique est alors :

\[
S_{INT4}
\approx
\frac{Q_{W,fp16}}{Q_{W,int4,total}}
=
\frac{2KM}{\frac{1}{2}KM(1+\eta)}
\]

Donc :

\[
S_{INT4}
=
\frac{4}{1+\eta}
\]

Si l’overhead des scales vaut 10% :

\[
\eta = 0.1
\]

alors :

\[
S_{INT4}
=
\frac{4}{1.1}
\approx
3.64
\]

Conclusion :

\[
S_{INT4} \leq 4
\]

sur la lecture des poids, dans le cas idéal memory-bound.

En pratique, le speedup est inférieur si :

- la dequantization coûte cher ;
- le kernel n’utilise pas bien les Tensor Cores ;
- le batch est assez grand pour devenir compute-bound ;
- les scales sont mal layoutées ;
- les accès mémoire sont non coalescés.

---

## 10. Opération 4 — Attention IO-aware

L’attention standard est :

\[
S = QK^T
\]

\[
P = softmax(S)
\]

\[
O = PV
\]

avec :

\[
Q,K,V \in \mathbb{R}^{L \times d}
\]

Le coût arithmétique est :

\[
F_{attn}
=
\Theta(L^2d)
\]

Ce coût ne disparaît pas si l’attention reste exacte.

Le problème principal vient du fait que l’attention standard matérialise souvent :

\[
S \in \mathbb{R}^{L \times L}
\]

et parfois :

\[
P \in \mathbb{R}^{L \times L}
\]

Donc le coût mémoire contient un terme :

\[
Q_{standard}
=
\Theta(L^2)
\]

en plus des tenseurs \( Q,K,V,O \).

### 10.1 Attention tiled

Une attention IO-aware découpe \( Q,K,V \) en blocs et calcule la softmax de manière incrémentale, sans matérialiser toute la matrice \( L \times L \).

On peut écrire de manière simplifiée :

\[
Q_{tiled}
=
\Theta(Ld)
+
\Theta\left(\frac{L^2d^2}{M}\right)
\]

où \( M \) est la taille disponible de mémoire rapide dans le modèle théorique.

Le speedup memory-bound devient :

\[
S_{attn}
\approx
\frac{Q_{standard}}{Q_{tiled}}
\]

Si :

\[
Q_{standard} \gg Q_{tiled}
\]

alors le gain peut être important.

Mais si l’opération devient compute-bound :

\[
\frac{F_{attn}}{P_h}
\geq
\frac{Q_{tiled}}{\beta_h}
\]

alors le speedup plafonne.

Conclusion : pour l’attention exacte, on ne réduit pas l’ordre arithmétique \( \Theta(L^2d) \), mais on peut réduire fortement le trafic mémoire.

---

## 11. Synthèse des speedups théoriques

| Bloc | Baseline | Optimisation | Borne théorique indicative |
|---|---|---|---:|
| RMSNorm | opérations séparées | fusion en un kernel | \(1.5\times\) à \(2\times\) |
| RoPE | kernel séparé + relecture | fusion avec attention/KV cache | jusqu’à \(3\times\) sur le sous-graphe |
| Matmul decode | poids FP16 | poids INT4 + dequant fusionnée | jusqu’à \(4/(1+\eta)\) |
| Attention | matérialisation \(L^2\) | tiling IO-aware | dépend de \(L,d,M,\beta_h,P_h\) |
| Dispatcher adaptatif | kernel fixe | meilleur kernel parmi candidats | non-pire si fallback + overhead faible |

---

## 12. Algorithme proposé

### Algorithm 1 — Adaptive Kernel Selection

```text
Input:
  operation o
  hardware h
  shape s
  dtype τ
  candidate kernels K_o

Output:
  selected kernel k*

1. if cache contains (o, h, s, τ):
2.     return cached kernel
3.
4. best_time ← +∞
5. best_kernel ← baseline
6.
7. for k in K_o:
8.     validate numerical correctness of k
9.     benchmark k on warmup + repeated runs
10.    measure median latency
11.    if latency < best_time:
12.        best_time ← latency
13.        best_kernel ← k
14.
15. cache[(o,h,s,τ)] ← best_kernel
16. return best_kernel
```

---

## 13. Preuve de non-régression

Comme le kernel baseline appartient à l’ensemble des candidats :

\[
k_{base} \in \mathcal{K}_o
\]

et comme le sélecteur prend le minimum :

\[
T_{k^\*}
=
\min_{k \in \mathcal{K}_o} T_k
\]

on a nécessairement :

\[
T_{k^\*}
\leq
T_{k_{base}}
\]

Donc le runtime adaptatif ne peut pas être plus lent que le baseline au niveau du kernel sélectionné.

Avec overhead :

\[
T_{AKO}
=
T_{k^\*} + \delta
\]

Donc :

\[
T_{AKO}
\leq
T_{base} + \delta
\]

Si \( \delta \) est petit devant \( T_{base} \), alors la non-régression est pratiquement garantie.

La condition expérimentale à vérifier est :

\[
\delta < \alpha T_{base}
\]

avec par exemple :

\[
\alpha = 0.01
\]

---

## 14. Exemple chiffré estimatif

Cette section donne des **valeurs estimées** pour visualiser l’ordre de grandeur. Ce ne sont pas des mesures réelles. Elles servent uniquement à vérifier que les formules sont cohérentes.

### 14.1 Hypothèses générales

On prend un scénario Transformer/LLM typique :

| Paramètre | Valeur estimée |
|---|---:|
| Batch size \(B\) | 8 |
| Sequence length \(L\) | 2048 |
| Hidden dimension \(D\) | 4096 |
| Nombre de tokens effectifs \(N = B \times L\) | 16 384 |
| Dtype | FP16/BF16 |
| Taille par élément \(b\) | 2 bytes |
| Bande passante effective illustrative \(\beta_h\) | 1.2 TB/s |

On utilise une bande passante effective illustrative de 1.2 TB/s, volontairement inférieure à un pic théorique possible, pour représenter une exécution réelle imparfaite.

---

### 14.2 Exemple RMSNorm

On a :

\[
N = B \times L = 8 \times 2048 = 16384
\]

\[
D = 4096
\]

\[
ND = 16384 \times 4096 = 67\,108\,864
\]

Le volume mémoire minimal pour lire ou écrire un tenseur de taille \(N \times D\) en FP16/BF16 est :

\[
ND b = 67\,108\,864 \times 2
= 134\,217\,728 \text{ bytes}
\]

Soit environ :

\[
128 \text{ MiB}
\]

#### Baseline

Avec :

\[
Q_{base} \approx 6ND b
\]

on obtient :

\[
Q_{base}
= 6 \times 128 \text{ MiB}
= 768 \text{ MiB}
\]

Temps estimé sous hypothèse memory-bound :

\[
T_{base}
\approx
\frac{768 \text{ MiB}}{1.2 \text{ TB/s}}
\]

En bytes décimaux approximatifs :

\[
T_{base}
\approx
0.67 \text{ ms}
\]

#### Kernel fusionné

Avec :

\[
Q_{fused} \approx 3ND b
\]

on obtient :

\[
Q_{fused}
= 3 \times 128 \text{ MiB}
= 384 \text{ MiB}
\]

Temps estimé :

\[
T_{fused}
\approx
\frac{384 \text{ MiB}}{1.2 \text{ TB/s}}
\approx
0.34 \text{ ms}
\]

Speedup estimé :

\[
S
=
\frac{0.67}{0.34}
\approx
1.97
\]

Conclusion estimative :

| Variante RMSNorm | Trafic estimé | Temps estimé | Speedup |
|---|---:|---:|---:|
| Baseline non fusionnée | 768 MiB | 0.67 ms | 1.00× |
| Kernel fusionné idéal | 384 MiB | 0.34 ms | 1.97× |
| Kernel fusionné avec relecture partielle | 512 MiB | 0.45 ms | 1.49× |

Ces valeurs collent avec la borne théorique :

\[
1.5 \leq S_{RMSNorm} \leq 2
\]

---

### 14.3 Exemple RoPE fusionné

On prend :

| Paramètre | Valeur |
|---|---:|
| Batch \(B\) | 8 |
| Sequence length \(L\) | 2048 |
| Nombre de têtes \(H\) | 32 |
| Dimension par tête \(d\) | 128 |
| Dtype | FP16/BF16 |
| Bytes par élément \(b\) | 2 |

Le volume de \(Q\) et \(K\) est :

\[
X = 2BLHd b
\]

\[
X = 2 \times 8 \times 2048 \times 32 \times 128 \times 2
\]

\[
X = 268\,435\,456 \text{ bytes}
\]

Soit environ :

\[
256 \text{ MiB}
\]

#### Baseline séparée

La baseline lit \(Q,K\), écrit \(Q',K'\), puis l’attention relit \(Q',K'\) :

\[
Q_{base} \approx 3X
\]

\[
Q_{base} \approx 768 \text{ MiB}
\]

Temps estimé :

\[
T_{base}
\approx
\frac{768 \text{ MiB}}{1.2 \text{ TB/s}}
\approx
0.67 \text{ ms}
\]

#### RoPE fusionné

La version fusionnée lit \(Q,K\) une fois, applique la rotation en registers, puis consomme directement :

\[
Q_{fused} \approx X
\]

\[
Q_{fused} \approx 256 \text{ MiB}
\]

Temps estimé :

\[
T_{fused}
\approx
\frac{256 \text{ MiB}}{1.2 \text{ TB/s}}
\approx
0.22 \text{ ms}
\]

Speedup estimé :

\[
S
=
\frac{0.67}{0.22}
\approx
3.0
\]

Tableau :

| Variante RoPE | Trafic estimé | Temps estimé | Speedup |
|---|---:|---:|---:|
| RoPE séparé + relecture | 768 MiB | 0.67 ms | 1.00× |
| RoPE fusionné | 256 MiB | 0.22 ms | 3.00× |
| RoPE fusionné avec overhead réel | 300 MiB équiv. | 0.25 ms | 2.68× |

Conclusion : le gain local peut approcher \(3\times\), mais le gain global modèle sera inférieur car RoPE n’est qu’un sous-graphe.

---

### 14.4 Exemple matmul INT4 en decode

On considère une couche linéaire :

\[
Y = XW
\]

avec :

\[
K = 4096
\]

\[
M = 4096
\]

Donc :

\[
KM = 4096 \times 4096 = 16\,777\,216
\]

#### Poids FP16

\[
Q_{W,fp16} = 2KM
\]

\[
Q_{W,fp16}
= 33\,554\,432 \text{ bytes}
\]

Soit environ :

\[
32 \text{ MiB}
\]

#### Poids INT4

\[
Q_{W,int4} = \frac{1}{2}KM
\]

\[
Q_{W,int4}
= 8\,388\,608 \text{ bytes}
\]

Soit environ :

\[
8 \text{ MiB}
\]

Avec un overhead de scales de \(\eta = 0.1\) :

\[
Q_{W,int4,total}
= 8 \text{ MiB} \times 1.1
= 8.8 \text{ MiB}
\]

Speedup théorique memory-bound :

\[
S_{INT4}
=
\frac{32}{8.8}
\approx
3.64
\]

Tableau :

| Variante matmul decode | Trafic poids estimé | Speedup local théorique |
|---|---:|---:|
| FP16 | 32 MiB | 1.00× |
| INT8 | 16 MiB + scales | environ 1.8× à 2.0× |
| INT4 | 8.8 MiB avec scales | environ 3.64× |

Attention : cette estimation concerne surtout la lecture des poids. Le gain réel dépendra du coût de dequantization et de l’efficacité du kernel.

---

### 14.5 Exemple attention IO-aware

On prend une attention avec :

| Paramètre | Valeur |
|---|---:|
| Sequence length \(L\) | 2048 |
| Dimension par tête \(d\) | 128 |
| Dtype | FP16/BF16 |
| Bytes par élément | 2 |

La matrice d’attention complète \(S\) a taille :

\[
L \times L = 2048 \times 2048 = 4\,194\,304
\]

En FP16 :

\[
4\,194\,304 \times 2
= 8\,388\,608 \text{ bytes}
\]

Soit environ :

\[
8 \text{ MiB}
\]

Si une baseline matérialise \(S\) puis \(P\), on a déjà :

\[
Q_{S,P} \approx 2 \times 8 \text{ MiB}
= 16 \text{ MiB}
\]

par tête, sans compter les relectures, \(Q,K,V,O\), et les accès additionnels.

Avec \(H=32\) têtes :

\[
16 \text{ MiB} \times 32 = 512 \text{ MiB}
\]

Une attention tiled évite de matérialiser entièrement \(S\) et \(P\), ce qui peut économiser plusieurs centaines de MiB de trafic mémoire pour ce cas.

Exemple estimatif :

| Variante attention | Trafic additionnel lié à \(S,P\) | Temps mémoire estimé à 1.2 TB/s |
|---|---:|---:|
| Baseline matérialisant \(S,P\) | 512 MiB | 0.45 ms |
| Attention tiled | 80–200 MiB équiv. | 0.07–0.17 ms |

Speedup local estimé sur cette partie :

\[
S_{attn,mem}
\approx
\frac{512}{200}
=2.56
\]

ou, dans un cas plus favorable :

\[
S_{attn,mem}
\approx
\frac{512}{80}
=6.4
\]

Mais le speedup global est plafonné par le coût arithmétique \(\Theta(L^2d)\). On ne doit donc pas annoncer un speedup global \(6\times\) sans mesure.

---

### 14.6 Exemple d’impact global modèle

Supposons que, dans une passe d’inférence, la répartition du temps soit :

| Composant | Part du temps baseline |
|---|---:|
| Matmul / Linear | 55% |
| Attention | 25% |
| RMSNorm | 8% |
| RoPE | 4% |
| Autres opérations | 8% |

Supposons les speedups locaux suivants :

| Composant | Speedup local estimé |
|---|---:|
| Matmul INT4/dequant fused | 1.6× |
| Attention IO-aware ou kernel spécialisé | 1.4× |
| RMSNorm fused | 1.7× |
| RoPE fused | 2.5× |
| Autres opérations | 1.0× |

Le speedup global se calcule avec la loi d’Amdahl :

\[
S_{global}
=
\frac{1}{
\sum_i \frac{p_i}{S_i}
}
\]

Donc :

\[
S_{global}
=
\frac{1}{
\frac{0.55}{1.6}
+
\frac{0.25}{1.4}
+
\frac{0.08}{1.7}
+
\frac{0.04}{2.5}
+
\frac{0.08}{1.0}
}
\]

Calcul :

\[
S_{global}
=
\frac{1}{
0.34375
+0.17857
+0.04706
+0.016
+0.08
}
\]

\[
S_{global}
=
\frac{1}{0.66538}
\approx
1.50
\]

Conclusion : même avec des speedups locaux de \(2\times\) ou \(3\times\), le speedup global réaliste peut être autour de \(1.3\times\) à \(1.8\times\), selon la part réelle des opérations optimisées.

Tableau final estimatif :

| Scénario | Hypothèse | Speedup global estimé |
|---|---|---:|
| Conservateur | petits gains, peu d’opérations couvertes | 1.10×–1.25× |
| Réaliste | kernels fusionnés sur zones chaudes | 1.30×–1.70× |
| Optimiste | matmul quantifié + attention optimisée + fusion | 1.80×–2.50× |
| Très optimiste | workload très spécifique, fortement memory-bound | >2.50× localement, pas garanti globalement |

---

## 15. Protocole expérimental proposé

Pour valider le papier, il faut mesurer :

- latence p50 ;
- latence p95 ;
- tokens/s ;
- GB/s effectifs ;
- TFLOP/s effectifs ;
- mémoire max ;
- erreur numérique max ;
- erreur relative ;
- overhead de compilation/autotuning ;
- cache hit ratio.

Pour chaque opération :

```text
RMSNorm
RoPE
RMSNorm + residual
RoPE + attention input
INT4 dequant + matmul
attention prefill
attention decode
```

Pour chaque shape :

```text
batch ∈ {1, 4, 8, 16}
seq_len ∈ {512, 2048, 4096, 8192}
hidden_dim ∈ {1024, 2048, 4096, 8192}
dtype ∈ {fp16, bf16, int8, int4}
```

Pour chaque GPU :

```text
L40S
A100
H100
RTX 4090
```

Dans un contexte industriel plus réaliste, il faut commencer avec seulement 1 ou 2 GPUs disponibles, par exemple L40S et A100.

---

## 16. Critères de validation numérique

Un kernel optimisé est accepté si :

\[
\text{max\_abs\_error}
<
\epsilon_{abs}
\]

et :

\[
\text{max\_rel\_error}
<
\epsilon_{rel}
\]

Par exemple en FP16/BF16 :

\[
\epsilon_{abs} = 10^{-2}
\]

\[
\epsilon_{rel} = 10^{-2}
\]

Pour les logits d’un modèle complet, on peut mesurer :

\[
\Delta_{logits}
=
\frac{\|Z_{base} - Z_{opt}\|_2}{\|Z_{base}\|_2}
\]

et imposer :

\[
\Delta_{logits} < \tau
\]

avec par exemple :

\[
\tau = 10^{-2}
\]

Pour un modèle de vision ou de détection, on mesure aussi :

- mAP50 ;
- mAP50-95 ;
- accuracy ;
- F1 ;
- latence inference ;
- mémoire VRAM.

---

## 17. Hypothèse principale du papier

L’hypothèse de recherche peut être formulée ainsi :

> Pour des workloads IA fixes, une sélection adaptative de kernels spécialisés par hardware, shape et dtype réduit le temps d’exécution par rapport à une baseline générique, sans perte numérique significative, lorsque les opérations ciblées sont memory-bound ou fusionnables.

Formellement :

\[
\exists k^\* \in \mathcal{K}_o
\quad
\text{tel que}
\quad
Q_{k^\*}(h,s) < Q_{base}(h,s)
\]

et si :

\[
\frac{Q_{base}}{\beta_h}
>
\frac{F_o}{P_h}
\]

alors :

\[
T_{k^\*}(h,s) < T_{base}(h,s)
\]

à overhead constant négligeable près.

---

## 18. Limites

Ce cadre ne prouve pas que l’on battra toujours TensorRT-LLM, FlashAttention, cuBLAS ou CUTLASS.

Il prouve plutôt :

- que la sélection adaptative ne peut pas être pire si le fallback est inclus ;
- que la fusion réduit mathématiquement le trafic mémoire ;
- que les gains sont bornés par le modèle roofline ;
- que les speedups existent surtout en régime memory-bound ;
- que le coût d’autotuning est amortissable.

Les gains réels dépendront de :

- GPU exact ;
- driver CUDA ;
- version PyTorch/Triton ;
- layout mémoire ;
- shape ;
- dtype ;
- batching ;
- niveau de fusion ;
- qualité du kernel ;
- occupancy ;
- pression sur les registres ;
- pression sur la shared memory.

---

## 19. Conclusion

Le projet est mathématiquement défendable si on le présente comme :

> un runtime adaptatif de kernels bas niveau, optimisé par réduction du trafic mémoire et sélection automatique selon le hardware.

Le résultat théorique principal est :

\[
T_{AKO}(h,s)
=
\min_{k \in \mathcal{K}_o} T_k(h,s)
+
\delta
\]

avec :

\[
T_k(h,s)
\geq
\max\left(
\frac{F_o(s)}{P_h},
\frac{Q_k(h,s)}{\beta_h}
\right)
+
\lambda_k
\]

Donc le projet est solide si l’on montre expérimentalement que pour certains blocs :

\[
Q_{opt}(h,s) < Q_{base}(h,s)
\]

et que l’opération est suffisamment memory-bound pour que cette réduction se traduise en baisse réelle de latence.

La contribution la plus crédible n’est pas :

> écrire du code assembleur plus rapide que tout le monde.

Mais plutôt :

> construire une couche adaptative, mesurable et reproductible, capable de sélectionner automatiquement le meilleur kernel pour une opération donnée, un GPU donné et une contrainte numérique donnée.

C’est un sujet sérieux, mathématiquement propre, et compatible avec un vrai projet de recherche appliquée ou MLOps avancé.

---

## 20. Références utiles

Ces références servent à situer le projet dans l’écosystème actuel.

1. **FlashAttention** — Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.  
   https://arxiv.org/abs/2205.14135

2. **FlashAttention IO complexity / follow-up literature** — travaux sur l’optimalité IO de l’attention.  
   https://arxiv.org/abs/2402.07443

3. **Triton autotune documentation** — mécanisme d’autotuning de kernels.  
   https://triton-lang.org/main/python-api/generated/triton.autotune.html

4. **PyTorch custom C++/CUDA operators** — documentation pour intégrer des kernels custom.  
   https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

5. **CUTLASS** — bibliothèque NVIDIA pour GEMM et primitives CUDA haute performance.  
   https://docs.nvidia.com/cutlass/index.html


