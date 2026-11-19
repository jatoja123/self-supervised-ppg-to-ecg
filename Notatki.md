## TODO
- zbadać jak ilość danych wpływa na długość treningu (logować też czas epoki i pretreningu)
- jak ilość danych i epok wpływa na jakość pretreningu?
- zbadać jak batch size wpływa na jakość uczenia
- sprawdzić na innych metrykach: po 2 nowe metody walidacji, plus te przez nich używane
- Kuba: poznać metryki np fretchet distance, dynamic time warping (są w prezentacji)
- Kuba: które parametry mogą mieć największy wpływ - dogadać - zakresy wartości np strata KL-divergence / architektura
- Ja: jak przetestować zrobione modele na nowych metrykach walidacyjnych? Wczytać, uruchomić, zapisać do wandb, unikać duplikowania 
- Ja: przygotować do puszczenia wielu eksperymentów na raz - jak?
- Ja: analiza wyników
- Zaimplementować dodatkowe metryki walidacyjne
- Zmienić rodzaj upsamplowania PPG? zgodnie z ich publikacją
- Stały zbiór testowy: z tego co zostało bierzemy jakiś % do treningu

### Pomysły co dalej
- spróbować dodać inne dane

## Co zrobiłem:
- niektóre pliki nie miały rozszerzenia .py, poprawiłem
- train.py pokazywał ładowanie danych jako losowe okna PPG/EKG, bez wczytywania ich znikąd
- pobrałem dane who-is-alyx, zrobiłem ich wczytywanie, matchowanie i podział na okna z pokryciem 20%
- chat wykrył błąd: zastąpił rekurencyjną odległość Frecheta wersją iteracyjną - ryzyko wywalenia się kodu
- Chat proponował mi dodanie l1_loss do straty generatora, ale nie jest to uwzględniane w publikacji - można sprawdzić wpływ. Ustawiłem na 0
- zreplikowałem pre-processing na podstawie artykułu, którego nie było - plik data.py pokazywał wczytywanie danych w innym podziale folderów/formacie i po preprocessingu
- dodałem wczytywanie konfiguracji z pliku YAML, przykładową konfigurację lite i przekopiowaną ichniejszą (base)
- możliwość zmniejszenia danych - data_fraction - wybierz tylko jakiś % uczestników
- dodałem zapisywanie metryk do W&B - przy uczeniu dyskryminatora oraz generatora
- dodatkowo zapisywanie konfiguracji startowej przy eksperymencie (kiedyś się na to naciąłem)
- dodałem zapisywanie najlepszego modelu po treningu

## Loss i metryki
- W pierwszym etapie SSL jest cross entropy loss bo model przyporządkowuje transformację sygnału do typu transformacji 0/1.
Dla mnie ryzykowna jest ta 'intuicja' w pre-trenowaniu modelu. Można sprawdzić czy to naprawdę pomaga, dla mniejszego modelu/danych.
- Walidacja to 10-fold cross-validation
- W drugim kroku generator ma Wasserstein loss czyli po prostu ocena dyskryminatora. Potencjalny problem: słaba odwzorowywalność spowrotem na PPG
- W drugim kroku dyskryminator ma Wassersterin loss czyli
D(fake) - D(real) + Gradient penalty (ograniczna nadmierne zmiany, wpsiera stabilność)
Czyli jeśli Dyskryminator ocenia Fake jako prawdziwy lub real jako fałszywy to duży loss.
- Podczas treningu oprócz start walidowano 10% danych
- Metryki testów: RMSE; Pearson; Frechet Distance; MAE HR - wyznaczanie tętna z fake/real sygnału EKG
Intuicja:
RMSE - dokładność rekonstrukcji
Pearson - zgodność skali, amplitudy
Fretchet D - podobny kształt krzywych
MAE HR - poprawność fizjologiczna

## Triki i Techniki
- Resampling: sprowadzanie próbek z różnych danych do wspólnej częstotliwości 
### W generatorze UNet+Transformer
* Batch Normalization	dla CNN encodera, bottlenecka i decodera
* Layer Normalization	dla Transformera
* Learned positional embedding dla Transformera - czyli embeddujemy same pozycje w sekwencji
* Dropout (0.1) w Attention i MLP Transformera
* Skip connections w U-Net - łączy dane stare i nowe
* Residual connections - sumuje wejście i wyjście (liczy się poprawka wejścia)
* Funkcja aktywacji GELU dla CNN i Transformera
* Funkcja aktywacji Tanh dla wyjścia Generatora
* Early stopping patience=12, z użyciem L1_loss na walidacji (ale nie ma tego w Loss_fn)
* Czyli bezwzględna różnica między sygnałami. Jeśli nasz sygnał zbyt się rozjedzie (w dowolnej osi) - zatrzymujemy uczenie.
* NIE ma żadnego gradient penalty np l1/l2
* Strata Wasserstein - TYLKO ocena dyskryminatora
### W dyskryminatorze - CNN + Transformer
* Dyskryminator — CNN + Transformer
* Layer Normalization	w CNN i Transformerze (nie używa BatchNorm)
* Dropout (0.1) w Attention i MLP Transformera
* Residual connections w Transformerze
* Gradient Penalty dla WGAN-GP
* Self Supervised Learning - nauka cech EKG
* Transfer learning + freezing (zamrożenie warstw) - Po etapie SSL
* Strata wasserstain - różnica między oceną real a fake. Ciekawe, że sama wartość nie ma znaczenia, tylko różnica między dwoma.