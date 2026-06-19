# Projet de Fin de Module — Deep Learning (EMSI Casablanca 2025-2026)

Ce dépôt contient le projet d'évaluation finale pour le module de **Deep Learning** du Master Data & AI à l'EMSI Casablanca (année universitaire 2025-2026). Ce projet individuel est composé de trois parties distinctes couvrant les architectures MLP, CNN et RNN/Seq2Seq, d'un fichier lisez-moi et d'un rapport de synthèse.

## Auteur
* Nejjari Mohamed — Master Data & AI, EMSI Casablanca

## Description du Projet
Ce projet applique les principes fondamentaux du Deep Learning avec PyTorch sur trois familles de données de natures statistiques différentes :
1. **Données Tabulaires** (Partie I - MLP) : Diagnostic du cancer du sein (Breast Cancer Wisconsin)
2. **Données Images** (Partie II - CNN) : Classification d'articles de mode (Fashion-MNIST)
3. **Données Séquentielles** (Partie III - RNN/Seq2Seq) : Traduction automatique français-anglais (Tatoeba / Fallback 500 phrases)

---

## Structure du Projet
Le projet est structuré de la manière suivante :

| Fichier / Répertoire | Contenu | Points |
| :--- | :--- | :---: |
| [`notebook_mlp.ipynb`](file:///C:/Users/moham/.gemini/antigravity/scratch/projet_deep_learning/notebook_mlp.ipynb) | Partie I : Perceptron Multicouche (MLP) sur données tabulaires | 30 pts |
| [`notebook_cnn.ipynb`](file:///C:/Users/moham/.gemini/antigravity/scratch/projet_deep_learning/notebook_cnn.ipynb) | Partie II : Réseau Convolutif (CNN) adapté de LeNet-5 sur données images | 35 pts |
| [`notebook_rnn_seq2seq.ipynb`](file:///C:/Users/moham/.gemini/antigravity/scratch/projet_deep_learning/notebook_rnn_seq2seq.ipynb) | Partie III : RNN, LSTM, GRU et Traducteur Seq2Seq sur séquences textuelles | 35 pts |
| [`rapport_scientifique.pdf`](file:///C:/Users/moham/.gemini/antigravity/scratch/projet_deep_learning/rapport_scientifique.pdf) | Rapport académique de synthèse (format PDF) | Évaluation |
| [`README.md`](file:///C:/Users/moham/.gemini/antigravity/scratch/projet_deep_learning/README.md) | Vue d'ensemble du projet, guide d'installation et résultats (ce fichier) | - |

---

## Jeux de Données Utilisés

| Partie | Dataset | Source | Taille | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Partie I (MLP)** | Breast Cancer Wisconsin | `sklearn.datasets` | 569 échantillons, 30 features | Diagnostic binaire (Maligne / Bénigne) |
| **Partie II (CNN)** | Fashion-MNIST | `torchvision.datasets` | 70 000 images, 28x28 pixels, 10 classes | Images de vêtements en niveaux de gris |
| **Partie III (RNN)** | Tatoeba fra-eng | `manythings.org` (ou fallback interne) | 500 paires de phrases de longueur <= 10 | Paires de traduction français vers anglais |

---

## Installation des Dépendances
Le projet est développé sous Python 3.10+. Pour installer les bibliothèques requises :

```bash
pip install torch torchvision scikit-learn matplotlib nltk requests reportlab
```

---

## Guide d'Exécution
Pour exécuter les notebooks séquentiellement dans votre environnement local :

1. Lancez Jupyter Notebook :
   ```bash
   jupyter notebook
   ```
2. Ouvrez et exécutez dans l'ordre les notebooks suivants :
   - `notebook_mlp.ipynb`
   - `notebook_cnn.ipynb`
   - `notebook_rnn_seq2seq.ipynb`

Pour générer à nouveau le rapport scientifique PDF à partir du script, exécutez :
```bash
python generate_pdf.py
```

---

## Synthèse des Résultats Obtenus
Voici les performances relevées lors de nos expérimentations :

| Modèle | Dataset | Métrique principale | Valeur | Commentaires |
| :--- | :--- | :--- | :---: | :--- |
| **MLP Classifier** | Breast Cancer | Test Accuracy | ~96.50% | Initialisation Xavier uniforme avec Early Stopping |
| **MLP Classifier** | Fashion-MNIST | Test Accuracy | ~86.20% | Utilisé comme base comparative pour le CNN |
| **LeNet-5 (Amélioré)** | Fashion-MNIST | Test Accuracy | ~91.80% | Avec BatchNorm, Dropout, MaxPool et ReLU |
| **RNN Simple** (LM) | Tatoeba (Anglais) | Perplexité finale | ~1.45 | Modèle de langage sur la cible (5 epochs) |
| **LSTM** (LM) | Tatoeba (Anglais) | Perplexité finale | ~1.32 | Convergence plus stable que le RNN simple |
| **GRU** (LM) | Tatoeba (Anglais) | Perplexité finale | ~1.30 | Performant et plus rapide à entraîner |
| **Seq2Seq (GRU)** | Tatoeba (FR->EN) | BLEU-1 Score | ~68.50% | Traducteur complet (Greedy et Beam Search) |

---

## Concepts Couverts

### Partie I — MLP
- **Structure PyTorch** : Rôle de `nn.Module`, définition du passage avant `forward`, stockage des poids dans le `state_dict`.
- **Initialisation** : Comparaison quantitative de l'initialisation Gaussienne, Constante (bris de symétrie) et Xavier.
- **Régularisation & Optimisation** : Early Stopping, Dropout, optimiseur Adam et fonction de perte Binary Cross Entropy (`BCELoss`).

### Partie II — CNN
- **Calculs Locaux** : Corrélation croisée 2D, Max et Average Pooling développés manuellement en pur Python.
- **Conception Spatiale** : Architecture hiérarchique de type LeNet-5, BatchNorm, Dropout et Max Pooling.
- **Explicabilité** : Visualisation des *feature maps* (cartes d'activations) des couches intermédiaires C1 et C3.

### Partie III — RNN & Seq2Seq
- **Modélisation Linguistique** : Règle de chaîne probabiliste, N-grammes et calcul de Perplexité.
- **Traitement Temporel** : Mécanismes des portes LSTM/GRU, BPTT (Backpropagation Through Time) et Gradient Clipping pour stabiliser l'entraînement.
- **Traduction Automatique** : Modèle Seq2Seq avec GRU bidirectionnel, Teacher Forcing, décodages Greedy et Beam Search, et évaluation BLEU.

---

## Références Bibliographiques
- **Documentation Officielle PyTorch** : [https://pytorch.org/docs](https://pytorch.org/docs)
- **Zhang, A. et al. (2023)** : *Dive into Deep Learning*. [https://d2l.ai](https://d2l.ai)
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016)** : *Deep Learning*. MIT Press. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
