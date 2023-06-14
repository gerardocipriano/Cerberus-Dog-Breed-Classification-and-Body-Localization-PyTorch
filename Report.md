# Relazione Laboratorio di Ottimizzazione, Intelligenza Artificiale e Machine Learning

# Cerberus

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="introduzione">Introduzione</a>
    </li>
    <li>
      <a href="#classificazione">Classificazione Multiclasse</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#architettura">Architettura</a></li>
        <li><a href="#addestramento">Addestramento</a></li>
        <li><a href="#valutazione">Valutazione</a></li>
      </ul>
    </li>
    <li>
      <a href="#localizzazione">Localizzazione</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#architettura">Architettura</a></li>
        <li><a href="#addestramento">Addestramento</a></li>
        <li><a href="#valutazione">Valutazione</a></li>
      </ul>
    </li>
  </ol>
</details>

## Introduzione

Il progetto in esame si chiama Cerberus e il suo obiettivo è creare un programma in grado di classificare correttamente 3 razze di cani e riconoscere 3 parti del corpo del cane. Per raggiungere questo obiettivo, è stata creata una GUI per consentire all’utente l’accesso a tutte le funzionalità di addestramento e riconoscimento.
Il codice utilizza principalmente le librerie PyTorch e torchvision per implementare il modello di classificazione delle immagini e per gestire i dati di addestramento. Inoltre, viene utilizzata la libreria TensorBoard per monitorare le prestazioni del modello durante l’addestramento.

## Classificazione Multiclasse

### Scelta del dataset

Il codice utilizza due dataset per addestrare il modello: il dataset Stanford Dog e il dataset Kaggle Breeds Cat. Questi due dataset sono stati scelti perché contengono un gran numero di immagini di cani e gatti di diverse razze, il che li rende adatti per addestrare un modello di classificazione delle immagini in grado di riconoscere le diverse razze di cani e gatti.

Per gestire i dati di addestramento, il codice utilizza due DataLoader personalizzati. Un DataLoader è un’interfaccia fornita dalla libreria PyTorch che consente di caricare i dati in modo efficiente durante l’addestramento del modello. In questo caso, i dati vengono suddivisi in tre set: train, validation e test. Il set di train viene utilizzato per addestrare il modello, il set di validation viene utilizzato per valutare le prestazioni del modello durante l’addestramento e il set di test viene utilizzato per valutare le prestazioni del modello dopo l’addestramento.

### Scelta dell’architettura

L’architettura scelta per il modello è AlexNet. Questa scelta è stata fatta perché AlexNet è una rete neurale convoluzionale profonda che ha dimostrato di essere molto efficace nella classificazione delle immagini. Inoltre, AlexNet è stata progettata per essere facilmente adattabile a nuovi compiti di classificazione, il che la rende una scelta ideale per questo progetto.

Per addestrare il modello, viene utilizzata la funzione di perdita CrossEntropyLoss e l’algoritmo di ottimizzazione SGD (Stochastic Gradient Descent) con un tasso di apprendimento e un momento specificati nella configurazione. La funzione di perdita CrossEntropyLoss è una scelta comune per i compiti di classificazione multiclasse perché misura la distanza tra le previsioni del modello e le etichette vere. L’algoritmo SGD è un metodo efficace per ottimizzare i pesi del modello durante l’addestramento.

### Addestramento

Il modello viene addestrato in due fasi. Nella prima fase, viene eseguito il transfer learning sul dataset dei cani congelando i pesi del modello preaddestrato di AlexNet. In questo modo, il modello può imparare a riconoscere le diverse razze di cani senza dover riaddestrare tutti i pesi della rete.

Nella seconda fase, viene eseguito l’addestramento sul dataset dei gatti scongelando i pesi del modello. In questo modo, il modello può utilizzare ciò che ha imparato durante l’addestramento sulle razze dei cani anche per riconoscere le razze dei gatti.

Durante l’addestramento, vengono calcolate la perdita e l’accuratezza sul set di train e sul set di validation e vengono registrate su TensorBoard per consentire un facile monitoraggio delle prestazioni del modello durante l’addestramento; Inoltre, engono registrati su TensorBoard anche gli embeddings, il primo batch di immagini per ogni epoca e le matrici di confusione per il set di train e il set di validation.

### Valutazione

.....
.....
.....

## Localizzazione

### Dataset

......
......
......

### Architettura

......
......
......

### Addestramento

......
......
......

### Valutazione

......
......
......
