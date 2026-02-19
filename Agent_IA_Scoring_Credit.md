# 🏦 Agent IA — Système de Scoring de Crédit Intelligent

> **Système** : AI Credit Risk Engine v3.1  
> **Modèle sous-jacent** : Gradient Boosting Ensemble (XGBoost + LightGBM) + Réseau de neurones (MLP)  
> **Référentiel réglementaire** : Bâle III · Bank Al-Maghrib Circulaire 19/G/2002 · IFRS 9  
> **Date d'analyse** : 19 février 2026 — 09:15 UTC  
> **Institution** : Banque Centrale Marocaine / Système bancaire universel

---

## Table des Matières

1. [Architecture du Moteur de Scoring](#1-architecture-du-moteur-de-scoring)
2. [Méthodologie & Variables d'Analyse](#2-méthodologie--variables-danalyse)
3. [Dossier Client #001 — Mohamed A.](#3-dossier-client-001--mohamed-a)
4. [Dossier Client #002 — Sara B.](#4-dossier-client-002--sara-b)
5. [Dossier Client #003 — Karim D.](#5-dossier-client-003--karim-d)
6. [Dossier Client #004 — Nadia R.](#6-dossier-client-004--nadia-r)
7. [Dossier Client #005 — Hassan M.](#7-dossier-client-005--hassan-m)
8. [Tableau de Synthèse — Toutes Décisions](#8-tableau-de-synthèse--toutes-décisions)
9. [Système de Détection de Biais & Équité](#9-système-de-détection-de-biais--équité)
10. [Spécification Technique du Modèle IA](#10-spécification-technique-du-modèle-ia)
11. [Cadre Réglementaire & Conformité](#11-cadre-réglementaire--conformité)

---

## 1. Architecture du Moteur de Scoring

```
╔══════════════════════════════════════════════════════════════════════╗
║              AI CREDIT RISK ENGINE — Architecture Globale           ║
╠════════════════════════╦═════════════════════════════════════════════╣
║  ENTRÉES               ║  Données financières · Historique crédit   ║
║                        ║  Données comportementales · Données bureau ║
╠════════════════════════╬═════════════════════════════════════════════╣
║  PREPROCESSING         ║  Normalisation · Imputation · Encodage     ║
║                        ║  Détection outliers · Feature engineering  ║
╠════════════════════════╬═════════════════════════════════════════════╣
║  MODÈLES ML            ║  XGBoost (poids: 35%)                      ║
║  (Ensemble)            ║  LightGBM (poids: 35%)                     ║
║                        ║  MLP Neural Network (poids: 20%)           ║
║                        ║  Logistic Regression (poids: 10%)          ║
╠════════════════════════╬═════════════════════════════════════════════╣
║  SCORING               ║  Score de risque [0.0 → 1.0]               ║
║                        ║  0.0 = Risque nul · 1.0 = Défaut certain   ║
╠════════════════════════╬═════════════════════════════════════════════╣
║  DÉCISION AUTOMATIQUE  ║  ACCEPTÉ · REFUSÉ · INFO COMPLÉMENTAIRE    ║
╠════════════════════════╬═════════════════════════════════════════════╣
║  EXPLICABILITÉ         ║  SHAP Values · Top 3 facteurs explicatifs   ║
║                        ║  Rapport détaillé · Recours possible        ║
╚════════════════════════╩═════════════════════════════════════════════╝
```

### Seuils de Décision Automatique

```
Score de risque :

0.0 ══════════════════════════════════════════════════════ 1.0

[0.00 – 0.30]   ████████████  ACCEPTÉ       ← Faible risque
[0.31 – 0.55]   ░░░░░░░░░░░░  INFO REQUISE  ← Risque modéré
[0.56 – 1.00]   ▓▓▓▓▓▓▓▓▓▓▓▓  REFUSÉ        ← Risque élevé
```

---

## 2. Méthodologie & Variables d'Analyse

### 2.1 Variables Financières (Poids : 55 %)

| Variable | Description | Impact sur le Score |
|---|---|---|
| **Taux d'endettement** | (Dettes totales / Revenus annuels) × 100 | ↑↑ fort si >40 % |
| **Ratio de couverture** | Revenus nets / Mensualité demandée | ↓↓ si <2.5 |
| **Capacité de remboursement** | Revenu disponible après charges fixes | ↓ si <30 % revenu |
| **Patrimoine net** | Actifs − Passifs | ↓ si négatif |
| **Stabilité des revenus** | Variance des revenus sur 24 mois | ↑ si haute variance |
| **Épargne moyenne** | Solde moyen compte épargne | ↓ si <3 mensualités |

### 2.2 Variables d'Historique Crédit (Poids : 30 %)

| Variable | Description | Impact |
|---|---|---|
| **Incidents de paiement** | Nombre de retards >30 jours (5 ans) | ↑↑ fort |
| **Défauts antérieurs** | Crédits non remboursés | ↑↑↑ critique |
| **Utilisation crédit revolving** | % du plafond utilisé | ↑ si >70 % |
| **Ancienneté crédit** | Durée depuis premier crédit | ↓ si <2 ans |
| **Mix crédit** | Diversité des types de crédit | ↓ si mono-type |
| **Requêtes récentes** | Demandes de crédit (12 mois) | ↑ si >3 |

### 2.3 Variables Contextuelles (Poids : 15 %)

| Variable | Description | Impact |
|---|---|---|
| **Stabilité emploi** | Ancienneté poste actuel | ↓ si <6 mois |
| **Secteur d'activité** | Risque sectoriel (CDI fonct. public vs interim) | Variable |
| **Âge du demandeur** | Durée active restante vs durée prêt | Modéré |
| **Situation familiale** | Charges familiales / personnes à charge | Modéré |
| **Région** | Dynamisme économique local | Faible |

---

## 3. Dossier Client #001 — Mohamed A.

### 3.1 Données du Demandeur

| Champ | Valeur |
|---|---|
| **Référence dossier** | CRED-2026-001-MA |
| **Type de crédit demandé** | Prêt immobilier |
| **Montant demandé** | 850 000 MAD |
| **Durée** | 20 ans |
| **Mensualité calculée** | 5 200 MAD |
| **Âge** | 38 ans |
| **Situation professionnelle** | Ingénieur — CDI secteur privé (ancienneté : 9 ans) |
| **Revenu net mensuel** | 18 500 MAD |
| **Charges mensuelles fixes** | 3 200 MAD (loyer actuel + charges) |
| **Épargne disponible** | 120 000 MAD |
| **Apport personnel** | 170 000 MAD (20 % du bien) |
| **Taux d'endettement actuel** | 17 % |
| **Incidents de paiement (5 ans)** | 0 |
| **Crédits en cours** | 1 crédit auto (mensualité : 1 200 MAD) |
| **Score bureau de crédit** | 742 / 850 |

### 3.2 Résultat du Scoring IA

```
╔═══════════════════════════════════════════════════════════════╗
║                  SCORE DE RISQUE — DOSSIER #001               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   Score de risque :   0.18 / 1.00                             ║
║                                                               ║
║   ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  18 %        ║
║   [FAIBLE RISQUE]                                             ║
║                                                               ║
║   Probabilité de défaut (PD) :       3.2 %                   ║
║   Perte en cas de défaut (LGD) :    41.0 %                   ║
║   Exposition en cas de défaut :   850 000 MAD                 ║
║   Perte attendue (EL) :            11 220 MAD                 ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   DÉCISION :   ✅  CRÉDIT ACCORDÉ                             ║
║                                                               ║
║   Taux proposé : 4,85 % fixe sur 20 ans                       ║
║   Mensualité : 5 200 MAD / mois                               ║
║   Assurance obligatoire : 180 MAD / mois                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 3.3 Trois Facteurs Explicatifs Principaux (SHAP)

#### 🟢 Facteur 1 — Stabilité professionnelle excellente (impact : −0.142 sur le risque)
Mohamed occupe un poste d'ingénieur en CDI depuis **9 ans** dans le secteur privé structuré. Cette longévité professionnelle réduit significativement le risque d'interruption de revenus. Le modèle identifie une corrélation inverse forte entre l'ancienneté >7 ans et le taux de défaut observé sur l'historique de données bancaires.

#### 🟢 Facteur 2 — Taux d'endettement post-crédit très acceptable (impact : −0.118 sur le risque)
Après ajout des mensualités du prêt immobilier (5 200 MAD) et du crédit auto (1 200 MAD), le taux d'endettement atteint **34,6 %** des revenus nets. Ce niveau reste en dessous du seuil critique de 40 % fixé par Bank Al-Maghrib, laissant une marge de sécurité de 1 000 MAD/mois.

#### 🟢 Facteur 3 — Historique de crédit irréprochable et apport significatif (impact : −0.097 sur le risque)
Zéro incident de paiement sur 5 ans, score bureau de crédit de 742/850 (centile 89), et apport personnel de 170 000 MAD (20 % du bien). L'apport réduit le loan-to-value (LTV) à 80 %, limitant l'exposition de la banque en cas de saisie immobilière.

---

## 4. Dossier Client #002 — Sara B.

### 4.1 Données du Demandeur

| Champ | Valeur |
|---|---|
| **Référence dossier** | CRED-2026-002-SB |
| **Type de crédit demandé** | Crédit à la consommation |
| **Montant demandé** | 80 000 MAD |
| **Durée** | 5 ans |
| **Mensualité calculée** | 1 580 MAD |
| **Âge** | 29 ans |
| **Situation professionnelle** | Enseignante — Fonctionnaire (ancienneté : 4 ans) |
| **Revenu net mensuel** | 6 200 MAD |
| **Charges mensuelles fixes** | 1 800 MAD |
| **Épargne disponible** | 15 000 MAD |
| **Taux d'endettement actuel** | 29 % |
| **Incidents de paiement (5 ans)** | 1 retard de 15 jours (résolu) |
| **Crédits en cours** | Aucun |
| **Score bureau de crédit** | 618 / 850 |

### 4.2 Résultat du Scoring IA

```
╔═══════════════════════════════════════════════════════════════╗
║                  SCORE DE RISQUE — DOSSIER #002               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   Score de risque :   0.27 / 1.00                             ║
║                                                               ║
║   ████████████████████████████░░░░░░░░░░░░░░░░░░  27 %        ║
║   [FAIBLE-MODÉRÉ]                                             ║
║                                                               ║
║   Probabilité de défaut (PD) :       5.8 %                   ║
║   Perte attendue (EL) :             1 894 MAD                 ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   DÉCISION :   ✅  CRÉDIT ACCORDÉ (conditions ajustées)       ║
║                                                               ║
║   Taux proposé : 7,20 % (prime de risque légère)              ║
║   Mensualité : 1 580 MAD / mois                               ║
║   Taux d'endettement post-crédit : 54,5 % ⚠️                  ║
║   → Montant recommandé revu à 60 000 MAD (mensualité 1 185 MAD)║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 4.3 Trois Facteurs Explicatifs Principaux (SHAP)

#### 🟢 Facteur 1 — Statut fonctionnaire = garantie de revenu (impact : −0.201 sur le risque)
Le statut d'enseignante fonctionnaire de l'État constitue la garantie de revenu la plus solide du système. L'impossibilité légale de licenciement et la prévisibilité totale des revenus réduisent massivement le risque d'interruption de paiement. Le modèle pondère ce facteur comme le plus protecteur dans la catégorie "stabilité emploi".

#### 🟡 Facteur 2 — Taux d'endettement post-crédit en zone limite (impact : +0.134 sur le risque)
L'ajout de la mensualité de 1 580 MAD porterait le taux d'endettement à **54,5 %** (3 380 MAD de charges / 6 200 MAD revenus), dépassant largement le seuil réglementaire de 40 %. Le modèle recommande de limiter le montant accordé à 60 000 MAD pour revenir à un taux d'endettement de 48 %, ou d'étendre la durée à 7 ans.

#### 🟡 Facteur 3 — Épargne insuffisante comme filet de sécurité (impact : +0.089 sur le risque)
L'épargne disponible de 15 000 MAD représente seulement **9,5 mois** de mensualités, en dessous du coussin de sécurité recommandé de 12 mois. En cas d'imprévu (maladie, réparation urgente), le risque de défaut à court terme augmente. Recommandation : constituer 3 mois de mensualités supplémentaires avant déblocage.

---

## 5. Dossier Client #003 — Karim D.

### 5.1 Données du Demandeur

| Champ | Valeur |
|---|---|
| **Référence dossier** | CRED-2026-003-KD |
| **Type de crédit demandé** | Prêt personnel |
| **Montant demandé** | 150 000 MAD |
| **Durée** | 7 ans |
| **Mensualité calculée** | 2 400 MAD |
| **Âge** | 45 ans |
| **Situation professionnelle** | Auto-entrepreneur (ancienneté : 2 ans) |
| **Revenu net mensuel déclaré** | 12 000 MAD (variable) |
| **Charges mensuelles fixes** | 5 200 MAD (3 crédits en cours) |
| **Épargne disponible** | 8 000 MAD |
| **Taux d'endettement actuel** | 43 % |
| **Incidents de paiement (5 ans)** | 3 retards >30 jours dont 1 en 2025 |
| **Crédits en cours** | 3 crédits actifs |
| **Score bureau de crédit** | 421 / 850 |
| **Variance revenus (24 mois)** | ±38 % |

### 5.2 Résultat du Scoring IA

```
╔═══════════════════════════════════════════════════════════════╗
║                  SCORE DE RISQUE — DOSSIER #003               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   Score de risque :   0.74 / 1.00                             ║
║                                                               ║
║   ████████████████████████████████████████████░░░░  74 %      ║
║   [RISQUE ÉLEVÉ]                                              ║
║                                                               ║
║   Probabilité de défaut (PD) :      28,4 %                   ║
║   Perte en cas de défaut (LGD) :    62,0 %                   ║
║   Perte attendue (EL) :            26 477 MAD                 ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   DÉCISION :   ❌  CRÉDIT REFUSÉ                              ║
║                                                               ║
║   Motif principal : Cumul de facteurs de risque critiques     ║
║   Recours possible : Oui — dans 6 mois avec plan d'assainissem║
║   Contact conseiller : agence@banque.ma · 0522-XXXXXX         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 5.3 Trois Facteurs Explicatifs Principaux (SHAP)

#### 🔴 Facteur 1 — Taux d'endettement critique avant même le nouveau crédit (impact : +0.289 sur le risque)
Avec 3 crédits en cours représentant 5 200 MAD de mensualités, le taux d'endettement actuel atteint déjà **43,3 %**, au-delà du plafond réglementaire de 40 %. L'ajout du nouveau crédit porterait ce taux à **62,7 %**, exposant le client à un risque de surendettement cliniquement documenté par les données historiques du bureau de crédit.

#### 🔴 Facteur 2 — Historique de défauts récents et récurrents (impact : +0.247 sur le risque)
Trois incidents de paiement >30 jours en 5 ans, dont **un en 2025** (moins de 12 mois), signalent un comportement de paiement dégradé et récent. Le modèle accorde un poids doublé aux incidents récents car ils reflètent l'état financier actuel du client, pas son passé lointain. Le score bureau de 421/850 (centile 18) confirme cette fragilité.

#### 🔴 Facteur 3 — Revenus d'auto-entrepreneur instables avec ancienneté insuffisante (impact : +0.198 sur le risque)
Une variance de revenus de ±38 % sur 24 mois combinée à une ancienneté de seulement 2 ans en tant qu'auto-entrepreneur constitue un profil de risque élevé. En l'absence de justificatifs de 3 années complètes d'activité et d'une tendance claire à la hausse des revenus, le modèle ne peut pas établir une capacité de remboursement fiable.

---

## 6. Dossier Client #004 — Nadia R.

### 6.1 Données du Demandeur

| Champ | Valeur |
|---|---|
| **Référence dossier** | CRED-2026-004-NR |
| **Type de crédit demandé** | Crédit immobilier |
| **Montant demandé** | 600 000 MAD |
| **Durée** | 25 ans |
| **Mensualité calculée** | 3 450 MAD |
| **Âge** | 34 ans |
| **Situation professionnelle** | Médecin libéral (ancienneté : 3 ans) |
| **Revenu net mensuel** | 22 000 MAD (en croissance) |
| **Charges mensuelles fixes** | 2 100 MAD |
| **Épargne disponible** | 85 000 MAD |
| **Apport personnel** | 120 000 MAD (20 %) |
| **Taux d'endettement actuel** | 9,5 % |
| **Incidents de paiement (5 ans)** | 0 |
| **Historique crédit** | Limité — 1 seul crédit étudiant soldé |
| **Score bureau de crédit** | 697 / 850 |
| **Données complémentaires** | Revenus non salariaux — variation saisonnière ±20 % |

### 6.2 Résultat du Scoring IA

```
╔═══════════════════════════════════════════════════════════════╗
║                  SCORE DE RISQUE — DOSSIER #004               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   Score de risque :   0.38 / 1.00                             ║
║                                                               ║
║   ████████████████████████████████████░░░░░░░░░░  38 %        ║
║   [RISQUE MODÉRÉ — ZONE D'INCERTITUDE]                        ║
║                                                               ║
║   Probabilité de défaut (PD) :       9,7 %                   ║
║   Confiance du modèle :              67 %  ← faible           ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   DÉCISION :   📋  INFORMATIONS COMPLÉMENTAIRES REQUISES      ║
║                                                               ║
║   Documents demandés :                                        ║
║   1. Bilans comptables certifiés 3 dernières années           ║
║   2. Attestation ordre des médecins + patente                 ║
║   3. Relevés bancaires professionnels 12 mois                 ║
║   4. Déclarations fiscales (IR) 2023 et 2024                  ║
║   Délai de réponse client : 15 jours ouvrables                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 6.3 Trois Facteurs Explicatifs Principaux (SHAP)

#### 🟢 Facteur 1 — Profil financier intrinsèquement solide (impact : −0.198 sur le risque)
Un taux d'endettement actuel de 9,5 %, une épargne de 85 000 MAD, un apport de 20 % et des revenus en croissance constante (+18 % sur 2 ans) composent un profil de solvabilité objectivement fort. Le taux d'endettement post-crédit de **25,2 %** est largement dans les normes bancaires.

#### 🟡 Facteur 2 — Historique de crédit insuffisant pour le modèle (impact : +0.176 sur le risque)
Un seul crédit antérieur (étudiant, soldé) ne permet pas au modèle d'établir un pattern de comportement de remboursement robuste sur des montants significatifs. Le score de 697/850 reflète non pas un mauvais comportement mais un **manque d'historique**. Le modèle ne peut pas distinguer un bon payeur d'un payeur inconnu avec le niveau de confiance requis pour un crédit de 600 000 MAD.

#### 🟡 Facteur 3 — Revenus libéraux non documentés suffisamment (impact : +0.154 sur le risque)
La nature libérale des revenus (±20 % de variation saisonnière) et l'ancienneté de 3 ans en cabinet propre nécessitent une vérification documentaire approfondie. Le modèle a besoin des bilans certifiés pour calculer le revenu moyen lissé réel (et non le revenu déclaré ponctuel) qui sert de base au calcul de capacité de remboursement sur 25 ans.

---

## 7. Dossier Client #005 — Hassan M.

### 7.1 Données du Demandeur

| Champ | Valeur |
|---|---|
| **Référence dossier** | CRED-2026-005-HM |
| **Type de crédit demandé** | Crédit professionnel (TPE) |
| **Montant demandé** | 300 000 MAD |
| **Durée** | 5 ans |
| **Mensualité calculée** | 6 100 MAD |
| **Âge** | 52 ans |
| **Situation professionnelle** | Gérant SARL — secteur commerce (ancienneté : 14 ans) |
| **Chiffre d'affaires annuel** | 1 800 000 MAD |
| **Résultat net annuel** | 145 000 MAD |
| **Charges mensuelles perso** | 4 200 MAD |
| **Épargne personnelle** | 210 000 MAD |
| **Garantie proposée** | Hypothèque sur local commercial (valeur 480 000 MAD) |
| **Taux d'endettement actuel** | 38 % |
| **Incidents de paiement (5 ans)** | 1 retard de 45 jours en 2022 (contexte Covid) |
| **Score bureau de crédit** | 658 / 850 |

### 7.2 Résultat du Scoring IA

```
╔═══════════════════════════════════════════════════════════════╗
║                  SCORE DE RISQUE — DOSSIER #005               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   Score de risque :   0.41 / 1.00                             ║
║                                                               ║
║   ████████████████████████████████████████░░░░░░  41 %        ║
║   [RISQUE MODÉRÉ — LIMITE HAUTE]                              ║
║                                                               ║
║   Probabilité de défaut (PD) :      11,3 %                   ║
║   Perte en cas de défaut (LGD) :    28 %  (garantie réduit)  ║
║   Perte attendue nette :           9 492 MAD                  ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   DÉCISION :   📋  INFORMATIONS COMPLÉMENTAIRES REQUISES      ║
║                                                               ║
║   Documents demandés :                                        ║
║   1. Bilan comptable certifié + liasse fiscale 2024           ║
║   2. Extrait RC + statuts SARL à jour                         ║
║   3. Évaluation indépendante du local commercial              ║
║   4. Prévisionnel activité 2026–2028                          ║
║   Note : Si documents OK → Probable accord avec taux 6,5 %    ║
║   La garantie hypothécaire réduit significativement le risque  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 7.3 Trois Facteurs Explicatifs Principaux (SHAP)

#### 🟢 Facteur 1 — Ancienneté et solidité de l'entreprise (impact : −0.187 sur le risque)
14 ans d'existence de la SARL avec un chiffre d'affaires de 1,8M MAD signalent une entreprise établie, ayant traversé plusieurs cycles économiques (dont la crise Covid) sans défaillance structurelle. Le modèle identifie une forte corrélation entre l'ancienneté >10 ans d'une TPE et la résilience face aux chocs de trésorerie.

#### 🟡 Facteur 2 — Garantie hypothécaire = réducteur de risque majeur (impact : −0.156 sur le risque)
L'hypothèque proposée sur le local commercial (480 000 MAD) offre un taux de couverture de **160 %** du montant emprunté. En cas de défaut, la LGD tombe à 28 % contre 62 % sans garantie. Ce facteur fait basculer la décision de "refus probable" vers "information complémentaire", rendant le dossier potentiellement viable.

#### 🟡 Facteur 3 — Taux d'endettement en zone limite avec revenus professionnels à documenter (impact : +0.163 sur le risque)
Le taux d'endettement de 38 % (sans le nouveau crédit) atteint 55 % post-crédit, dépassant le seuil réglementaire. Cependant, pour les crédits professionnels, le calcul intègre la capacité bénéficiaire de l'entreprise (145k MAD/an = 12 083 MAD/mois), ce qui nécessite une documentation comptable certifiée pour être pris en compte par le moteur de décision.

---

## 8. Tableau de Synthèse — Toutes Décisions

| # | Client | Crédit demandé | Score IA | Décision | Taux proposé | Facteur clé |
|---|---|---|---|---|---|---|
| 001 | Mohamed A. | Immobilier 850k MAD | **0.18** | ✅ **ACCORDÉ** | 4,85 % fixe | Stabilité professionnelle + 0 incident |
| 002 | Sara B. | Conso 80k MAD | **0.27** | ✅ **ACCORDÉ** (60k MAD) | 7,20 % | Fonctionnaire + montant ajusté |
| 003 | Karim D. | Personnel 150k MAD | **0.74** | ❌ **REFUSÉ** | — | Surendettement + défauts récents |
| 004 | Nadia R. | Immobilier 600k MAD | **0.38** | 📋 **INFO REQUISE** | TBD | Revenus libéraux non documentés |
| 005 | Hassan M. | Pro 300k MAD | **0.41** | 📋 **INFO REQUISE** | ~6,5 % | Garantie forte mais endettement limite |

### Visualisation des Scores

```
Score de risque (0 = sans risque, 1 = défaut certain)

Mohamed A. (001) : ██░░░░░░░░░░░░░░░░░░  0.18  ✅ ACCORDÉ
Sara B.    (002) : ███░░░░░░░░░░░░░░░░░  0.27  ✅ ACCORDÉ
Nadia R.   (004) : ████░░░░░░░░░░░░░░░░  0.38  📋 INFO
Hassan M.  (005) : ████░░░░░░░░░░░░░░░░  0.41  📋 INFO
Karim D.   (003) : ████████░░░░░░░░░░░░  0.74  ❌ REFUSÉ

             ├─── Zone verte ──┼──── Zone orange ───┼─ Zone rouge ─┤
             0.0             0.30                 0.55            1.0
```

---

## 9. Système de Détection de Biais & Équité

### 9.1 Principes Anti-Discrimination

> Le moteur de scoring est audité trimestriellement pour détecter tout biais systémique contraire aux réglementations en vigueur.

| Variable exclue | Raison d'exclusion |
|---|---|
| **Genre** | Interdit — discrimination directe |
| **Origine ethnique / nationalité** | Interdit — discrimination directe |
| **Religion** | Interdit — discrimination directe |
| **Situation de handicap** | Interdit — discrimination directe |
| **Grossesse / situation familiale** | Interdit — proxy discriminatoire |
| **Code postal seul** | Proxy de discrimination ethnique potentiel |

### 9.2 Mécanismes de Contrôle des Biais

```
AUDIT BIAIS — PROCESSUS TRIMESTRIEL

1. Parité démographique    → Taux d'acceptation similaire entre groupes protégés
2. Équité des chances      → Même taux de faux négatifs entre groupes
3. Calibration             → PD calibrée identiquement pour tous les groupes
4. Test d'invariance       → Score identique si seul le genre/l'origine change
```

### 9.3 Résultats Dernière Audit (Janvier 2026)

| Métrique | Résultat | Seuil acceptable | Statut |
|---|---|---|---|
| Parité taux d'acceptation H/F | 0.97 (vs 1.0 idéal) | >0.80 | ✅ |
| Disparate Impact | 0.94 | >0.80 | ✅ |
| Equal Opportunity (vrais positifs) | 0.96 | >0.90 | ✅ |
| Biais géographique rural/urbain | 0.89 | >0.85 | ✅ |

---

## 10. Spécification Technique du Modèle IA

### 10.1 Modèles et Performance

```yaml
ensemble_model:
  composants:
    - name: XGBoost
      version: 2.0.1
      poids: 0.35
      hyperparameters:
        n_estimators: 800
        max_depth: 6
        learning_rate: 0.05
        subsample: 0.8

    - name: LightGBM
      version: 4.2
      poids: 0.35
      hyperparameters:
        num_leaves: 63
        learning_rate: 0.05
        n_estimators: 1000

    - name: MLP Neural Network
      poids: 0.20
      architecture: [128, 64, 32, 1]
      activation: relu
      dropout: 0.3

    - name: Logistic Regression
      poids: 0.10
      regularization: L2

performances:
  dataset_validation: 2.4M dossiers (2015-2025)
  AUC-ROC: 0.924
  Gini: 0.848
  KS-Statistic: 0.712
  Brier Score: 0.067

explicabilite:
  methode: SHAP (SHapley Additive exPlanations)
  niveau: Individuel par dossier
  top_facteurs: 3 (affiché au décideur humain)
  recours: Oui — procédure de révision manuelle disponible
```

### 10.2 Cycle de Vie du Modèle

| Phase | Fréquence | Description |
|---|---|---|
| Réentraînement | Trimestriel | Sur nouveaux dossiers + résultats à 12 mois |
| Validation | Mensuelle | Suivi AUC-ROC, Gini, dérive des distributions |
| Audit biais | Trimestriel | Test parité démographique + équité |
| Révision architecture | Annuelle | Ajout nouvelles variables si RGPD compatible |
| Rapport régulateur | Semestriel | Soumission Bank Al-Maghrib |

---

## 11. Cadre Réglementaire & Conformité

### 11.1 Textes Applicables

| Texte | Organisme | Application |
|---|---|---|
| **Circulaire 19/G/2002** | Bank Al-Maghrib | Classification et provisionnement des créances |
| **Loi 103-12** | Parlement marocain | Établissements de crédit et organismes assimilés |
| **Loi 09-08** | CNDP | Protection des données personnelles |
| **Bâle III** | Comité de Bâle | Calcul fonds propres réglementaires (PD, LGD, EAD) |
| **IFRS 9** | IASB | Provisionnement des pertes de crédit attendues (ECL) |

### 11.2 Droits du Demandeur

> Conformément à la loi 09-08 et aux pratiques de l'IA responsable :

- **Droit à l'explication** : Tout client peut demander une explication détaillée de la décision automatique
- **Droit au recours humain** : Toute décision automatique peut être réexaminée par un conseiller humain sous 5 jours ouvrables
- **Droit à la rectification** : Données inexactes peuvent être corrigées et le dossier réanalysé
- **Droit d'opposition** : Le client peut s'opposer à la décision automatique et demander une analyse 100 % humaine
- **Contact recours** : `credit-recours@banque.ma` · `0522-XXXXXX` · Délai traitement : 10 jours ouvrables

### 11.3 Limites de l'Automatisation

> ⚠️ **Avertissement important** : Ce système d'IA est un outil d'aide à la décision. Pour les montants >500 000 MAD ou les dossiers en zone "Information requise", la décision finale est soumise à validation par un analyste crédit humain. Le modèle ne se substitue jamais à la responsabilité humaine dans la décision d'octroi de crédit.

---

*Rapport généré par AI Credit Risk Engine v3.1 — 19 février 2026, 09:15 UTC*  
*Classification : Confidentiel — Usage interne banque uniquement*  
*Prochain audit du modèle : 15 avril 2026*  
*Contact équipe Data Science : datascience@banque.ma*
