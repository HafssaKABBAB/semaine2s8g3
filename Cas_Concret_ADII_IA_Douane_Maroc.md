# 🇲🇦 Cas Concret : L'Intelligence Artificielle au Service de la Douane Marocaine

## Projet ADII — IA, Analyse Prédictive et Lutte contre la Fraude Douanière (2024–2028)

> **Statut du projet** : En cours de déploiement  
> **Institution porteuse** : Administration des Douanes et Impôts Indirects (ADII)  
> **Tutelle** : Ministère de l'Économie et des Finances — Royaume du Maroc  
> **Horizon stratégique** : Plan Quinquennal ADII 2024–2028  
> **Partenaires** : Organisation Mondiale des Douanes (OMD) · SECO (Suisse) · MEF

---

## Table des Matières

1. [Présentation du Projet](#1-présentation-du-projet)
2. [Contexte et Genèse](#2-contexte-et-genèse)
3. [Architecture Technique du Projet](#3-architecture-technique-du-projet)
4. [Le Centre Régional de Télé-contrôle (CRT) de Tanger](#4-le-centre-régional-de-télé-contrôle-crt-de-tanger)
5. [Partenariat International : ADII × OMD × SECO](#5-partenariat-international--adii--omd--seco)
6. [Résultats et Impacts Observés](#6-résultats-et-impacts-observés)
7. [Défis et Limites Identifiés](#7-défis-et-limites-identifiés)
8. [Feuille de Route et Prochaines Étapes](#8-feuille-de-route-et-prochaines-étapes)
9. [Leçons pour la Politique Publique Marocaine](#9-leçons-pour-la-politique-publique-marocaine)
10. [Sources](#10-sources)

---

## 1. Présentation du Projet

### En une phrase

> L'ADII intègre l'Intelligence Artificielle dans ses processus de **ciblage douanier, d'analyse prédictive des risques et d'interprétation d'images de scanners** afin de lutter plus efficacement contre la fraude, la contrebande et le blanchiment d'argent, tout en accélérant le dédouanement des opérateurs commerciaux licites.

### Fiche d'identité

| Élément | Détail |
|---|---|
| **Nom du projet** | Intégration de l'IA dans la Gestion des Risques Douaniers |
| **Institution** | Administration des Douanes et Impôts Indirects (ADII) |
| **Lancement** | Août 2024 (CRT Tanger) — Mai 2025 (partenariat OMD-SECO) |
| **Périmètre géographique** | Tanger-Med (pilote) → Casablanca → National |
| **Budget estimé** | Non communiqué publiquement |
| **Bénéficiaires directs** | Opérateurs économiques, transitaires, transporteurs |
| **Enjeu fiscal** | Recettes douanières (TVA, droits d'importation) |

---

## 2. Contexte et Genèse

### 2.1 Le défi structurel de la douane marocaine

L'ADII gère chaque année des **millions de déclarations en douane** (Déclarations Uniques de Marchandises — DUM) traitées via le système informatique **BADR** (*Base Automatisée des Douanes en Réseau*). Face à l'explosion du commerce international — notamment via le port de **Tanger-Med**, premier port d'Afrique et de la Méditerranée — les méthodes de contrôle traditionnelles atteignaient leurs limites :

- Volume de déclarations impossible à contrôler manuellement à 100 %
- Fraude de plus en plus sophistiquée (prix de transfert, admission temporaire détournée, faux manifestes)
- Pression sur les délais de dédouanement nuisant à la compétitivité des opérateurs honnêtes
- Ressources humaines limitées face à une croissance continue des flux

### 2.2 Le tournant stratégique de 2024

Le **Plan Quinquennal ADII 2024-2028**, publié sur le portail du MEF, marque un tournant en plaçant l'intelligence artificielle au cœur de la stratégie de modernisation. L'ADII s'engage explicitement à :

- Exploiter l'IA pour **l'interprétation des images de scanners**
- Élaborer des **modèles d'analyse prédictive** basés sur les données des DUM
- Mettre en place un **dispositif de renseignement intégré** combinant IA et bases de données dédiées
- Déployer une solution de **tracking RFID** des conteneurs et ensembles routiers

---

## 3. Architecture Technique du Projet

### 3.1 Les quatre composantes technologiques

```
┌─────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE IA — ADII 2024-2028                   │
├──────────────────┬──────────────────┬───────────────────────────┤
│  ANALYSE         │  VISION          │  TRACKING                 │
│  PRÉDICTIVE      │  ARTIFICIELLE    │  INTELLIGENT              │
│                  │                  │                           │
│  Modèles ML      │  Interprétation  │  Scellés électroniques    │
│  sur données DUM │  images scanners │  + RFID conteneurs        │
│  Scoring risque  │  Détection       │  Suivi temps réel         │
│  fraude par      │  anomalies       │  opérations de transit    │
│  déclaration     │  marchandises    │                           │
├──────────────────┴──────────────────┴───────────────────────────┤
│                  BASE DE DONNÉES CENTRALE IA                    │
│         Centralisation · Analyse · Apprentissage continu        │
├─────────────────────────────────────────────────────────────────┤
│                  SYSTÈME BADR (infrastructure existante)        │
│        Base Automatisée des Douanes en Réseau — socle SI        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Le ciblage algorithmique des déclarations

Le système de **scoring IA** analyse chaque DUM selon une multitude de paramètres et oriente les déclarations vers l'un des circuits de contrôle :

| Circuit | Couleur | Signification | Action |
|---|---|---|---|
| **Circuit 1** | 🟢 Vert | Risque faible — opérateur fiable | Dédouanement automatique |
| **Circuit 2** | 🟠 Orange | Risque modéré — vérification documentaire | Contrôle au CRT à distance |
| **Circuit 3** | 🔴 Rouge | Risque élevé — anomalie détectée | Inspection physique sur site |

L'IA permet d'**affiner en continu** ce ciblage en apprenant des résultats des contrôles précédents (*machine learning supervisé*).

---

## 4. Le Centre Régional de Télé-contrôle (CRT) de Tanger

### 4.1 Mise en service — 1er août 2024

L'ADII a mis en service un Centre Régional de Télé-contrôle (CRT) à Tanger, à compter du 1er août 2024, destiné aux opérateurs économiques, transitaires agréés en douane, transporteurs et exploitants des Magasins et Aires de Dédouanement (MEAD).

### 4.2 Fonctionnement du CRT

Le CRT regroupe les opérations de télé-contrôle des déclarations en détail de la région nord, grâce à une équipe d'inspecteurs vérificateurs chargés du contrôle à distance. Cette avancée est rendue possible par la dématérialisation totale du circuit de dédouanement via le système informatique BADR.

Concrètement, le CRT permet de :

- Traiter **à distance** les déclarations d'importation de Tanger-Med (circuits vert et orange)
- Éliminer les déplacements physiques inutiles des opérateurs
- Concentrer les inspecteurs qualifiés sur un plateau centralisé doté d'outils IA
- Réduire les **délais de dédouanement** qui impactent directement la compétitivité portuaire

### 4.3 Plan de déploiement progressif

```
PHASE 1 (Août 2024)     →  CRT Tanger (Tanger-Med + MEAD Tanger)
         ↓
PHASE 2 (En cours)      →  Duplication à Casablanca (Port de Casablanca)
         ↓
PHASE 3 (Horizon 2026)  →  Centre National de Télé-contrôle (couverture nationale)
```

---

## 5. Partenariat International : ADII × OMD × SECO

### 5.1 Lancement officiel — 19 mai 2025, Rabat

Le 19 mai 2025 à Rabat, a été donné le coup d'envoi d'un projet visant à intégrer les technologies d'IA dans les processus d'analyse et de gestion des risques douaniers, en partenariat entre l'ADII, l'Organisation Mondiale des Douanes (OMD) et le Secrétariat d'État à l'économie de la Confédération suisse (SECO). Ce projet, axé sur le ciblage et l'analyse prédictive, s'inscrit pleinement dans les efforts de digitalisation avancée engagés par l'ADII.

### 5.2 Objectifs du partenariat tripartite

| Objectif | Description |
|---|---|
| **Évaluation des acquis** | Cartographie des capacités actuelles de l'ADII en matière de traitement automatisé des données |
| **Ciblage intelligent** | Développement de modèles prédictifs pour identifier les déclarations à risque |
| **Transfert de savoir-faire** | Partage des meilleures pratiques douanières mondiales (standards OMD) |
| **Plan d'action structuré** | Feuille de route pour les prochaines étapes de déploiement IA |

### 5.3 Valeur ajoutée du partenariat OMD

L'OMD apporte une expertise unique via ses **500 membres** (administrations douanières mondiales), notamment :

- Cadres normatifs pour l'analyse prédictive douanière
- Benchmarks internationaux de détection de fraude par IA
- Standards de protection des données dans les échanges douaniers
- Protocoles d'interopérabilité entre systèmes douaniers nationaux

---

## 6. Résultats et Impacts Observés

### 6.1 Résultats opérationnels documentés

#### Détection de fraude par analyse de données — Cas multinationales (Octobre 2025)

Le système informatique BADR a permis aux brigades de contrôle de remonter la piste des écarts entre les importations déclarées et les exportations effectives. Des signalements émis par les services de renseignement et d'analyse des risques de l'administration sont à l'origine d'une opération d'audit d'envergure ciblant trois multinationales soupçonnées d'avoir exploité le régime d'admission temporaire pour échapper au paiement de droits et taxes douanières estimés à plusieurs milliards de dirhams.

Cet exemple illustre la puissance de la **détection automatisée d'anomalies** : l'écart entre importations déclarées et exportations effectives, invisible lors d'un contrôle manuel standard, a été détecté algorithmiquement.

#### Capacités du dispositif de renseignement IA

Grâce à des outils d'analyse sophistiqués, l'ADII peut non seulement interpréter les images issues des scanners, mais aussi créer des modèles d'analyse prédictive. Ces capacités permettent de détecter les anomalies et les schémas de fraude avec une précision inédite, rendant possible une intervention rapide et ciblée.

### 6.2 Impacts attendus selon le Plan 2024-2028

| Indicateur | Situation avant IA | Objectif avec IA |
|---|---|---|
| Délai moyen de dédouanement | Plusieurs jours | Réduction significative |
| Taux de détection de fraude | Ciblage aléatoire | Ciblage prédictif précis |
| Déplacements opérateurs | Fréquents et coûteux | Quasi-éliminés (télé-contrôle) |
| Couverture des contrôles | Partielle (ressources limitées) | Élargie sans surcoût RH |
| Schémas frauduleux détectés | Tardifs | Anticipés (analyse prédictive) |

---

## 7. Défis et Limites Identifiés

### 7.1 🔐 Cybersécurité des données douanières

Le système BADR centralise des données commerciales ultra-sensibles (flux d'importation/exportation de milliers d'entreprises). Son intégration avec des modules IA élargit la surface d'attaque potentielle pour des cybermenaces. La protection de ces données constitue un **impératif de sécurité nationale**.

### 7.2 ⚖️ Risque de faux positifs algorithmiques

Un modèle mal calibré peut orienter en circuit rouge des opérateurs légitimes, engendrant des **retards injustifiés** et des pertes économiques. L'équilibre entre sensibilité (détecter les fraudes) et spécificité (ne pas pénaliser les honnêtes) est un défi technique permanent.

### 7.3 👨‍💼 Adaptation des ressources humaines

Le passage au télé-contrôle requiert une **requalification des inspecteurs** vers des compétences en analyse de données et interprétation d'alertes algorithmiques. La résistance au changement dans les administrations publiques est un facteur à gérer soigneusement.

### 7.4 📋 Cadre juridique incomplet

L'utilisation de l'IA pour prendre des décisions administratives (sélection de circuits de contrôle, déclenchement d'audits) soulève des questions de **responsabilité algorithmique** et de **recours** pour les opérateurs injustement ciblés, dans un vide juridique partiel.

### 7.5 🔄 Interopérabilité des systèmes

L'intégration des modules IA avec l'infrastructure BADR existante (développée progressivement depuis les années 2000) pose des défis techniques d'**interopérabilité** et de migration des données historiques.

---

## 8. Feuille de Route et Prochaines Étapes

### 8.1 Court terme (2025-2026)

- Duplication du CRT à **Casablanca** (port et aéroport Mohammed V)
- Déploiement opérationnel des **modèles d'analyse prédictive** sur les DUM
- Intégration des **résultats du partenariat OMD-SECO** dans le plan d'action IA
- Formation des inspecteurs du CRT à l'utilisation des outils IA

### 8.2 Moyen terme (2026-2028)

- Lancement du **Centre National de Télé-contrôle** (couverture de l'ensemble du territoire)
- Déploiement de la solution de **tracking RFID** des conteneurs et ensembles routiers
- Intégration de la **reconnaissance d'images de scanners** par IA dans tous les points de contrôle majeurs
- Interconnexion avec les systèmes douaniers des **partenaires commerciaux** (UE, pays africains, ZLECAf)

### 8.3 Long terme (2028-2030)

- Déploiement d'un **système de renseignement douanier intégré** alimenté en temps réel par l'IA
- Contribution au **pôle numérique régional arabo-africain** piloté par le Ministère de la Transition numérique
- Partage d'expérience avec les douanes africaines dans le cadre de la **ZLECAf**

---

## 9. Leçons pour la Politique Publique Marocaine

### 9.1 Ce que ce projet démontre ✅

Ce projet constitue un **modèle de déploiement progressif et raisonné** de l'IA dans l'administration publique marocaine. Ses enseignements sont précieux :

**Démarche pilote avant généralisation** — Le CRT de Tanger a été conçu comme expérimentation contrôlée avant duplication nationale. Cette approche réduit les risques d'échec à grande échelle.

**Adossement à une infrastructure existante** — L'IA est intégrée dans BADR (système éprouvé) et non déployée ex nihilo, garantissant la continuité opérationnelle.

**Ancrage dans un plan stratégique pluriannuel** — Le Plan 2024-2028 donne un cadre de gouvernance, des objectifs mesurables et une vision long terme, évitant la dispersion des efforts.

**Partenariat international pour le transfert de compétences** — L'association avec l'OMD et le SECO apporte légitimité, expertise et financement externe, accélérant la montée en compétences.

### 9.2 Ce qui reste à faire ⚠️

- Développer un **cadre juridique explicite** sur l'utilisation de l'IA dans les décisions douanières
- Publier des **indicateurs de performance** de l'IA (taux de détection, faux positifs, économies réalisées) pour renforcer la transparence
- Prévoir des **mécanismes de recours** accessibles pour les opérateurs injustement ciblés
- Assurer la **formation continue** des agents face à l'évolution rapide des outils IA

### 9.3 Réplicabilité vers d'autres administrations financières

| Administration | Application potentielle inspirée de l'ADII |
|---|---|
| **DGI** | Ciblage prédictif des contrôles fiscaux (sur modèle du ciblage douanier) |
| **TGR** | Détection d'anomalies dans les dépenses publiques (équivalent des anomalies DUM) |
| **Cour des Comptes** | Audit automatisé des comptes publics (équivalent du télé-contrôle) |
| **AMMC** | Surveillance des marchés financiers par analyse prédictive |

---

## 10. Sources

### Sources primaires

| Source | Type | Date |
|---|---|---|
| **Plan Quinquennal ADII 2024-2028** (MEF) | Document officiel | 2024 |
| **Communiqué ADII — Lancement CRT Tanger** | Communiqué officiel | 1er août 2024 |
| **Coup d'envoi projet IA ADII-OMD-SECO** | Communiqué officiel | 19 mai 2025 |
| **Page Wikipedia ADII** | Source encyclopédique | Mise à jour 2025 |

### Sources médias

| Média | Article | Date |
|---|---|---|
| *Le Matin* | "Comment le ministère des Finances intègre l'IA dans ses activités" | 6 août 2024 |
| *La Vie Éco* | "Douane : l'IA pour traquer la fraude et la contrebande" | 2024 |
| *La Vie Éco* | "ADII : un Centre Régional de télé-contrôle implanté à Tanger" | 2024 |
| *LesEco.ma* | "Tanger : l'ADII lance un Centre régional de télé-contrôle" | 2 août 2024 |
| *Maroc Diplomatique* | "L'ADII réinvente la lutte contre la fraude avec un dispositif de renseignement de pointe" | 26 août 2024 |
| *Bladi.net / Hespress* | "Une fraude massive découverte par la douane marocaine" | Octobre 2025 |
| *LeBrief* | "Quel futur pour l'IA au Maroc ?" | Janvier 2026 |

### Références institutionnelles complémentaires

- **Organisation Mondiale des Douanes (OMD)** — Cadres de gestion des risques douaniers
- **SECO** (Secrétariat d'État à l'économie, Suisse) — Coopération au développement numérique
- **Ministère de la Transition numérique** — Stratégie Digital Morocco 2030
- **MEF** — Portail des marchés publics et publications réglementaires

---

*Document généré le 19 février 2026 — Basé sur des sources vérifiées et actualisées.*  
*Ce document s'inscrit dans le prolongement de l'analyse générale : "Intelligence Artificielle et Politiques Publiques Financières au Maroc".*
