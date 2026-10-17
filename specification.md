# Instrukcja użytkownika projektu PPG → ECG

Dokument opisuje konfigurację i aktualne zachowanie kodu w repozytorium.
Głównym punktem wejścia jest `code/train.py`: przygotowuje dane, opcjonalnie
uczy rozpoznawania transformacji ECG (SSL), a następnie uczy generator PPG → ECG
przy użyciu WGAN-GP.

## 1. Konfiguracja środowiska

### Wymagania

- Python **3.14 lub nowszy**, zgodnie z `pyproject.toml`.
- Menedżer pakietów **uv**; repozytorium zawiera `uv.lock`.
- Dane PPG i ECG pobrane osobno.
- CPU wystarcza do sprawdzenia danych i testów. Do pełnego treningu zalecana
  jest karta NVIDIA oraz instalacja PyTorch obsługująca CUDA.

Wszystkie poniższe polecenia wykonuj w katalogu głównym repozytorium.
Przykłady używają PowerShell na Windows.

### Instalacja

Po zainstalowaniu `uv` wykonaj:

```powershell
uv sync --locked
```

Polecenie tworzy środowisko `.venv` i instaluje wersje zapisane w `uv.lock`.
Zależności zadeklarowane w `pyproject.toml`:

| Pakiet | Wymagana wersja | Zastosowanie |
| --- | --- | --- |
| `numpy` | `>=2.5.4` | Tablice i operacje na sygnałach |
| `scipy` | `>=1.18.1` | Filtry i interpolacja |
| `scipy-stubs` | `~=1.18.1` | Informacje o typach dla narzędzi programistycznych |
| `torch` | `>=2.14.1` | Modele i trening |

Te zależności wystarczają do głównej ścieżki treningu. Dodatkowe moduły analizy
używają pakietów spoza tej listy: `eval.py` wymaga `pandas`,
`baseline_analysis.py` wymaga `pandas` i `matplotlib`, a `af_detection.py`
wymaga `matplotlib` i `scikit-learn`. Funkcja rysowania transformacji ECG
wymaga `matplotlib`. Jeśli korzystasz z tych modułów, dodaj potrzebne pakiety,
np.:

```powershell
uv add pandas matplotlib scikit-learn
```

`uv add` aktualizuje także `pyproject.toml` i `uv.lock`.

### Interpreter w PyCharm

Otwórz katalog repozytorium jako projekt i ustaw interpreter na
`.venv\Scripts\python.exe`. Dla konfiguracji uruchomienia ustaw:

| Pole | Wartość przykładowa |
| --- | --- |
| Script path | `code/train.py` |
| Working directory | Katalog główny repozytorium |
| Parameters | `--sampling-rate 128 --check-data` |

Aktywowanie środowiska w terminalu nie jest konieczne, jeśli podajesz pełną
ścieżkę do jego interpretera, jak w przykładach poniżej. Na Linux/macOS
odpowiednikiem jest `.venv/bin/python`.

### CPU i CUDA

Domyślnie skrypt wybiera `cuda`, gdy `torch.cuda.is_available()` zwraca `True`,
a w pozostałych przypadkach `cpu`. Dostępność sprawdzisz poleceniem:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Opcja `--device cuda` wymusza użycie CUDA, ale nie instaluje jej obsługi.
Jeżeli powyższe sprawdzenie zwraca `False`, użyj `--device cpu` lub skonfiguruj
sterownik i dystrybucję PyTorch odpowiednią do swojej karty.

## 2. Przygotowanie danych Who is Alyx?

### Układ katalogów

Domyślny loader `alyx_data:load_alyx_segments` szuka danych najpierw w
`data/who-is-alyx`, jeśli taki katalog istnieje, a w przeciwnym razie w
`data/who-is-alyx-2.1`. Ścieżkę można wskazać przez `--data-dir`.

```text
data/
└── who-is-alyx-2.1/
    └── players/
        ├── 42/
        │   └── 2022-05-19/
        │       ├── polar-h10/
        │       │   └── ECG.json
        │       └── empatica/
        │           └── PPG.json
        ├── 43/
        │   └── ...
        └── 44/
            └── ...
```

`--data-dir` wskazuje katalog **zawierający `players`**, nie sam katalog
`players`. Identyfikatory uczestników w tym loaderze muszą być liczbami
całkowitymi. Do podziału train/validation/test potrzeba co najmniej trzech
uczestników z użytecznymi parami sygnałów.

Pobierz rzeczywiste pliki JSON. Pliki `ECG.json.dvc` i `PPG.json.dvc` są tylko
wskaźnikami. Jeśli korzystasz z kopii datasetu zarządzanej przez DVC, pobierz
dane poleceniem `dvc pull` wykonanym w katalogu datasetu, po skonfigurowaniu
DVC i dostępu do jego magazynu. DVC nie jest wymagane do treningu, jeśli
masz już komplet plików JSON.

### Obsługiwane formaty JSON

Oryginalne nagrania Alyx zawierają numery próbek i czasy odbioru:

```json
{
  "data": {
    "1": [-81, "2022-05-19 16:27:02.850967"],
    "2": [-69, "2022-05-19 16:27:02.851582"]
  }
}
```

Klucze `1`, `2`, … są indeksami próbek. Loader odtwarza czas przy użyciu
częstotliwości natywnych: **ECG 130 Hz**, **PPG 64 Hz**, z czasem odbioru
pierwszej próbki jako początkiem. Czasy odbioru opisują transmisję pakietów;
kilka próbek może mieć tę samą datę. Luki w indeksach pozostają lukami
czasowymi. Utraty próbek bez luk w numeracji oraz różnic opóźnienia transmisji
między urządzeniami nie da się skorygować na podstawie tych plików.

Loader obsługuje również format `"timestamp": [wartość]` lub
`"timestamp": wartość`, ze znacznikami Unix w s/ms/µs/ns albo datami ISO.
Próbki są sortowane, a z powtórzonych znaczników czasu pozostaje pierwsza
wartość. Daty bez strefy czasowej muszą być interpretowane we wspólnej strefie.

### Przetwarzanie sygnałów

Każda sesja przechodzi następujące kroki:

1. Odczyt odpowiadających sobie plików ECG i PPG.
2. Wyznaczenie wspólnego przedziału czasowego nagrań.
3. Interpolacja liniowa na wspólną siatkę `--sampling-rate`.
4. Podział przy przerwach większych niż trzy natywne okresy próbkowania:
   około 23,1 ms dla ECG i 46,9 ms dla PPG.
5. Filtrowanie ciągłych fragmentów: ECG filtrem FIR 0,5–45 Hz, 401
   współczynników; PPG filtrem Butterwortha 0,5–8 Hz, rzędu 4.
   Oba filtry używają `filtfilt`.
6. Podział na okna o długości `round(window_seconds * sampling_rate)`.
   Krok okna to `max(1, int(window_size * (1 - overlap)))` próbek.
7. Odrzucenie par, w których rozstęp ECG lub PPG wynosi najwyżej `1e-8`.
8. Niezależna normalizacja każdego okna każdego sygnału do `[-1, 1]`
   i konwersja do `float32`.

Okna nie przechodzą między sesjami ani przez wykryte duże przerwy. Małe luki
mogą zostać interpolowane. Natywne częstotliwości 130/64 Hz są stałe dla
loadera Alyx; `--sampling-rate` zmienia docelową siatkę obu sygnałów.

Filtr ECG wymaga co najmniej **1204 próbek ciągłego fragmentu**, niezależnie
od długości pojedynczego okna. To około 9,26 s przy 130 Hz lub 9,41 s przy
128 Hz. Fragment musi mieć także co najmniej długość wybranego okna.
Loader Alyx wymaga docelowej częstotliwości większej niż 90 Hz.

Normalizacja usuwa informację o bezwzględnej amplitudzie. Wynik generatora
jest sygnałem znormalizowanym, a nie automatycznie ECG w jednostkach fizycznych.

## 3. Pierwsze uruchomienie

### Długość okna a generator

Generator ma cztery etapy zmniejszania długości o połowę. Do treningu wybierz
liczbę próbek w oknie **podzielną przez 16**, aby długość wyjścia odpowiadała
długości wzorca ECG.

Aktualne wartości domyślne CLI, 130 Hz i 4 s, dają 520 próbek. Generator
zwraca dla nich 512 próbek, co powoduje niezgodność rozmiarów podczas
treningu. `--check-data` tego nie wykrywa, ponieważ nie uruchamia modelu.
Przykłady treningu poniżej używają **128 Hz i 4 s = 512 próbek**.
Alternatywą jest 130 Hz i 8 s = 1040 próbek, kosztem większego zużycia pamięci.

### Sprawdzenie danych bez treningu

```powershell
.\.venv\Scripts\python.exe code/train.py --sampling-rate 128 --check-data
```

Skrypt odczyta cały zbiór, wykona preprocessing i wypisze liczby okien oraz
uczestników w trzech podzbiorach. Ta operacja może potrwać i używa pamięci RAM.

### Krótkie uruchomienie próbne

```powershell
.\.venv\Scripts\python.exe code/train.py --sampling-rate 128 --batch-size 2 --epochs-pretrain 1 --epochs-gan 1 --device cpu
```

To nadal jedna pełna epoka każdego etapu na całym zbiorze, więc na CPU może
być czasochłonne. Szybkie testy techniczne opisano w sekcji 10.

### Pełny trening

```powershell
.\.venv\Scripts\python.exe code/train.py --sampling-rate 128 --batch-size 8 --epochs-pretrain 90 --epochs-gan 90
```

### Własna ścieżka i wybrany uczestnik testowy

```powershell
.\.venv\Scripts\python.exe code/train.py --data-dir "D:\datasets\who-is-alyx-2.1" --sampling-rate 128 --held-out-subject 44
```

### Trening bez etapu SSL

```powershell
.\.venv\Scripts\python.exe code/train.py --sampling-rate 128 --epochs-pretrain 0 --epochs-gan 90
```

## 4. Wszystkie opcje `train.py`

Pomoc: `.\.venv\Scripts\python.exe code/train.py --help`.

| Opcja | Domyślnie przy uruchomieniu CLI | Znaczenie |
| --- | --- | --- |
| `-h`, `--help` | — | Wyświetla pomoc i kończy działanie. |
| `--data-loader` | `alyx_data:load_alyx_segments` | Funkcja odczytu danych w formacie `moduł:funkcja`. |
| `--data-dir` | Wybór loadera | Katalog danych; dla Alyx katalog zawierający `players`. |
| `--sampling-rate` | `130` | Docelowa częstotliwość w Hz; liczba całkowita. Alyx wymaga `>90`. |
| `--window-seconds` | `4.0` | Dodatnia długość okna w sekundach. |
| `--overlap` | `0.2` | Ułamek nakładania okien, `0 <= overlap < 1`. `0.2` oznacza 20%. |
| `--held-out-subject` | Pierwszy ID z `np.unique(subject_ids)` | Uczestnik przeznaczony do testu; musi mieć użyteczne okna. |
| `--batch-size` | `8` | Dodatnia liczba okien w batchu, wspólna dla SSL i GAN. |
| `--epochs-pretrain` | `90` | Maksymalna liczba epok SSL; `0` pomija etap. |
| `--epochs-gan` | `90` | Maksymalna liczba epok GAN; `0` nie wykonuje aktualizacji GAN. |
| `--device` | `cuda` albo `cpu`, zależnie od dostępności CUDA | Urządzenie PyTorch, np. `cpu`, `cuda`, `cuda:0`. |
| `--seed` | `42` | Ziarno losowości treningu, transformacji SSL i podziału uczestników. |
| `--check-data` | Wyłączone | Przygotowuje dane i podział, po czym kończy działanie bez tworzenia modeli. |

`--epochs-gan 0` nadal prowadzi do końcowego przejścia generatora na batchu
testowym. Do samej kontroli danych używaj `--check-data`.

## 5. Podział danych i przebieg uczenia

Podział odbywa się po uczestnikach. Wszystkie sesje i nakładające się okna
jednego uczestnika trafiają do tego samego podzbioru:

- **test**: jeden uczestnik wskazany przez `--held-out-subject`;
- **validation**: losowo wybrane około 10% pozostałych uczestników,
  dokładnie `max(1, round(0.1 * liczba_pozostałych))`;
- **train**: pozostali uczestnicy.

Proporcje dotyczą uczestników, więc liczby okien nie muszą zachowywać tych
samych proporcji. Pojedyncze uruchomienie wykonuje jeden taki podział;
pełną walidację LOSO trzeba uruchomić osobno dla kolejnych ID testowych.

Etap SSL uczy dyskryminator rozpoznawania sześciu transformacji ECG.
Każdemu oknu odpowiada sześć przykładów generowanych na żądanie. Etykiety
w `TransformationRecognitionDataset` mają wartości 0–5. Przykłady są
deterministyczne dla indeksu i ziarna, także w kolejnych epokach.
Uczenie korzysta z ECG uczestników treningowych, a walidacja z uczestników
walidacyjnych.

Po SSL kod przenosi enkoder CNN, transformer, token CLS i embeddingi pozycji
do nowego dyskryminatora i **zamraża te parametry** (`freeze_backbone=True`
w `main()`). Głowa krytyka jest uczona w GAN. Przy pominięciu SSL backbone
dyskryminatora pozostaje trenowalny.

GAN wykonuje `n_critic` aktualizacji krytyka na tym samym batchu, a następnie
jedną aktualizację generatora. Funkcje celu:

```text
L_D = mean(D(fake)) - mean(D(real)) + lambda_gp * mean((norm(grad D) - 1)^2)
L_G = -mean(D(fake)) + lambda_l1 * mean(abs(fake - real))
```

SSL używa entropii krzyżowej. Early stopping monitoruje stratę walidacyjną
SSL albo walidacyjne L1 generatora. Po 12 epokach bez poprawy etap kończy się
wcześniej. Najlepsze wagi dyskryminatora SSL i generatora GAN są przywracane
w pamięci; dyskryminator GAN pozostaje z ostatniej wykonanej epoki.

## 6. Hiperparametry modeli i treningu

Ustawienia znajdują się w `ModelConfig` w `code/train.py`. Projekt nie ma
obecnie pliku konfiguracji YAML/JSON ani opcji CLI dla każdego pola.
Parametry spoza tabeli CLI zmienia się w kodzie, np. edytując wartości
domyślne dataclass lub argumenty konstruktora `cfg = ModelConfig(...)`
w `main()`.

**CLI tworzy nowy obiekt konfiguracji.** Sama zmiana globalnego `CFG` nie
przekazuje do niego wszystkich ustawień. Pola obsługiwane przez CLI są
nadpisywane wartościami parsera. Szczególnie `ModelConfig.batch_size` wynosi
128, ale uruchomienie CLI domyślnie używa **8**.

### Trening

| Pole `ModelConfig` | Domyślnie w dataclass | Znaczenie |
| --- | --- | --- |
| `batch_size` | `128` | Liczba przykładów w batchu; CLI nadpisuje na `8`. |
| `epochs_pretrain` | `90` | Limit epok SSL; dostępny w CLI. |
| `epochs_gan` | `90` | Limit epok GAN; dostępny w CLI. |
| `lr_generator` | `1e-4` | Learning rate optymalizatora Adam generatora. |
| `lr_discriminator` | `1e-4` | Learning rate Adam dyskryminatora w SSL i GAN. |
| `betas` | `(0.5, 0.9)` | Współczynniki momentów Adam dla obu modeli. |
| `lambda_gp` | `10.0` | Waga kary gradientowej krytyka. |
| `lambda_l1` | `10.0` | Waga rekonstrukcji L1 w stracie generatora. |
| `n_critic` | `5` | Liczba aktualizacji krytyka na aktualizację generatora. |
| `early_stopping_patience` | `12` | Liczba epok bez poprawy przed zakończeniem etapu. |
| `device` | Automatycznie `cuda` lub `cpu` | Urządzenie; dostępne w CLI. |

Przykładowo, aby zmienić tempo uczenia z CLI w obecnej implementacji,
zmień domyślne `lr_generator` i `lr_discriminator` w `ModelConfig`.
Opcje `--lr-generator`, `--lambda-l1` i `--n-critic` nie są zaimplementowane.

### Sygnał i generator

| Pole `ModelConfig` | Domyślnie | Znaczenie |
| --- | --- | --- |
| `sampling_rate` | `130` | Docelowa częstotliwość danych; dostępna w CLI. |
| `in_channels` | `1` | Liczba kanałów wejściowych generatora. Główny pipeline dostarcza jeden kanał. |
| `out_channels` | `1` | Liczba kanałów wyjścia; pipeline i krytyk oczekują jednego ECG. |
| `num_unet_layers` | `4` | Pole informacyjne: architektura ma cztery etapy zapisane jawnie; zmiana pola nie zmienia ich liczby. |
| `base_channels` | `8` | Kanały enkodera: `base`, `2*base`, `4*base`, `8*base`. |
| `gen_transformer_dim` | `16` | Wymiar tokenów transformera generatora. |
| `gen_transformer_layers` | `2` | Liczba bloków transformera generatora. |
| `gen_transformer_heads` | `2` | Liczba głów attention. |
| `gen_transformer_mlp_ratio` | `4` | Mnożnik wymiaru warstwy ukrytej MLP względem wymiaru tokenów. |
| `gen_transformer_dropout` | `0.1` | Dropout transformera generatora. |
| `max_tokens` | `4096` | Pojemność embeddingów pozycji generatora po downsamplingu. |

Generator korzysta z konwolucji 1D, połączeń skip i wyjścia `Tanh`.
Zmiana częstotliwości w konfiguracji modelu nie przelicza danych samodzielnie;
odpowiada za to loader.

### Dyskryminator

| Pole `ModelConfig` | Domyślnie | Znaczenie |
| --- | --- | --- |
| `disc_conv_channels` | `(64, 128, 256)` | Kanały trzech warstw CNN. |
| `disc_conv_kernels` | `(65, 33, 17)` | Rozmiary jąder tych konwolucji. |
| `disc_conv_stride` | `1` | Krok konwolucji CNN. |
| `disc_model_dim` | `256` | Wymiar tokenów transformera dyskryminatora. |
| `disc_transformer_layers` | `2` | Liczba bloków transformera. |
| `disc_transformer_heads` | `2` | Liczba głów attention. |
| `disc_ffn_dim` | `512` | Żądany wymiar MLP; kod używa mnożnika `disc_ffn_dim // disc_model_dim`. |
| `disc_dropout` | `0.1` | Dropout transformera. |
| `disc_max_tokens` | `8192` | Pojemność embeddingów pozycji, łącznie z tokenem CLS. |
| `ssl_hidden_dim1` | `128` | Pierwsza warstwa ukryta głowy SSL. |
| `ssl_hidden_dim2` | `128` | Druga warstwa ukryta głowy SSL. |
| `ssl_num_classes` | `6` | Liczba klas; odpowiada sześciu transformacjom używanym w treningu. |
| `finetune_hidden_dim1` | `512` | Pierwsza warstwa dodatkowej głowy `mode="finetune"`. |
| `finetune_hidden_dim2` | `256` | Druga warstwa tej głowy. |

Główna pętla GAN korzysta z `mode="critic"`, a SSL z `mode="ssl"`.
Głowa `finetune` istnieje w modelu, ale nie jest używana w tych pętlach.

### Zależności między ustawieniami

- Wymiar tokenów musi dzielić się przez liczbę głów attention, osobno dla
  generatora i dyskryminatora.
- Ostatnia wartość `disc_conv_channels` musi odpowiadać `disc_model_dim`;
  kod nie ma między nimi dodatkowej projekcji.
- Aby `disc_ffn_dim` odpowiadał rzeczywistemu wymiarowi MLP, wybieraj
  dodatnią wielokrotność `disc_model_dim`.
- Dla domyślnego `disc_conv_stride=1` wymagane jest
  `T + 1 <= disc_max_tokens`, gdzie `T` to liczba próbek okna.
  CLI sprawdza także `T // 16 <= max_tokens`.
- Przy czterech etapach generatora wybieraj `T >= 16` podzielne przez 16.
- Większy batch zwiększa zużycie pamięci. Wydłużenie okna szczególnie obciąża
  transformer dyskryminatora, który przy domyślnym kroku CNN widzi wszystkie
  próbki oraz CLS.

## 7. Hiperparametry transformacji SSL

Ustawienia są w `TransformationConfig` w
`code/self_supervised_transformation.py`. Nie mają opcji CLI.
Główna ścieżka treningu używa domyślnego obiektu konfiguracji transformacji.

| Transformacja | Pola i wartości domyślne | Działanie |
| --- | --- | --- |
| `noise_addition` | `snr_min=2.0`, `snr_max=45.0` | Szum Gaussa o losowanym SNR w dB. |
| `scaling` | `scale_min=0.1`, `scale_max=10.0` | Mnożenie amplitudy przez losowany współczynnik. |
| `temporal_inversion` | Brak | Odwrócenie kolejności próbek. |
| `permutation` | `segments_min=2`, `segments_max=40` | Podział na fragmenty i permutacja ich kolejności. |
| `negation` | Brak | Zmiana znaku amplitudy. |
| `time_warping` | `segments_min=2`, `segments_max=40`, `stretch_min=1.05`, `stretch_max=4.0` | Losowe rozciąganie/ściskanie fragmentów i przywrócenie długości okna. |
| Ograniczenie podziału | `min_segment_length=4` | Ogranicza liczbę fragmentów zależnie od długości wejścia. |

Po transformacji wartości mogą wykraczać poza `[-1, 1]`, np. przy skalowaniu.
W SSL nie są ponownie normalizowane.

Moduł transformacji zawiera także klasę `original` i pomocniczy słownik
etykiet 0–6. Główny trening korzysta z własnego `TransformationRecognitionDataset`
z sześcioma klasami 0–5, bez klasy `original`. Nie należy podmieniać tych
datasetów bez sprawdzenia etykiet i liczby wyjść głowy SSL.

## 8. Podłączenie innego zbioru danych

Utwórz np. `code/my_data.py` z funkcją:

```python
def load_segments(data_dir, *, sampling_rate, window_seconds, overlap_ratio):
    # Odczytaj i wyrównaj sygnały, wykonaj resampling i filtrację.
    # Utwórz pary okien i znormalizuj każde z nich do [-1, 1].
    # Zwróć przygotowane tablice opisane poniżej.
    return ppg_segments, ecg_segments, subject_ids
```

Kontrakt wyniku:

| Element | Wymagania |
| --- | --- |
| `ppg_segments` | Tablica `(N, T)`, skończone wartości w `[-1, 1]`. |
| `ecg_segments` | Identyczny kształt; ten sam wiersz odpowiada temu samemu przedziałowi co PPG. |
| `subject_ids` | Tablica `(N,)`, jeden identyfikator uczestnika na parę okien. |
| `N` | Większe od zera. |
| `T` | Co najmniej 16; do obecnego generatora wybierz wielokrotność 16. |

`load_training_data` konwertuje sygnały do `float32` i sprawdza ich kształt,
skończoność i zakres. Synchronizacja, filtrowanie i normalizacja należą do
loadera. `data_dir` jest obiektem `Path` albo `None`, jeśli użytkownik go nie
poda. ID mogą być liczbami lub napisami, spójnymi dla wszystkich sesji danego
uczestnika. Minimum to trzech uczestników.

Uruchomienie:

```powershell
.\.venv\Scripts\python.exe code/train.py --data-loader my_data:load_segments --data-dir data/my-dataset --sampling-rate 128 --window-seconds 4 --check-data
```

Po poprawnym sprawdzeniu usuń `--check-data`, aby rozpocząć trening.
Ograniczenia filtrów loadera Alyx nie obowiązują automatycznie własnego
loadera; nadal obowiązują ograniczenia architektury modelu.

## 9. Wyniki, zapis wag i ewaluacja

W konsoli pojawiają się liczby okien i ID uczestników, a podczas uczenia:

- `[SSL]`: strata i accuracy dla treningu oraz walidacji;
- `[GAN]`: strata krytyka, strata generatora i walidacyjne L1;
- na końcu: rozmiary wejściowego PPG i wygenerowanego ECG dla jednego batcha
  testowego.

**`train.py` obecnie nie zapisuje automatycznie wag, historii ani wygenerowanych
sygnałów na dysk.** Nie ma też opcji CLI wznowienia treningu ani katalogu
wynikowego. Zwracane przez `train_wgan_gp` modele i `history` pozostają
w pamięci `main()`.

Jeśli chcesz zachować wynik, dodaj np. poniższy kod w `main()`, bezpośrednio
po wywołaniu `train_wgan_gp(...)`:

```python
from dataclasses import asdict

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
torch.save({
    "generator": generator.state_dict(),
    "discriminator": discriminator.state_dict(),
    "config": asdict(cfg),
    "history": history,
    "arguments": {**vars(args), "data_dir": str(args.data_dir) if args.data_dir else None},
    "train_subjects": np.unique(subject_ids[final_train_idx]).tolist(),
    "validation_subjects": np.unique(subject_ids[val_idx]).tolist(),
    "test_subjects": np.unique(subject_ids[test_idx]).tolist(),
}, output_dir / "ppg_to_ecg.pt")
```

To przykład rozszerzenia kodu, a nie istniejąca funkcja CLI. Taki plik
przechowuje wagi i opis eksperymentu, ale nie stan optymalizatorów potrzebny
do dokładnego wznowienia uczenia. Kolejne uruchomienie nadpisze ten sam plik.

Końcowy batch testowy w `train.py` służy jedynie do sprawdzenia przejścia
modelu; skrypt nie oblicza metryk dla całego testu.

`code/eval.py` udostępnia `evaluate_dataset` dla rzeczywistego i wygenerowanego
ECG o kształcie `(N, T)`, opcjonalnego PPG i częstotliwości `fs`.
Oblicza m.in. RMSE, korelację Pearsona, dyskretną odległość Frécheta oraz
błąd oszacowania tętna. Aby ocenić własny model, najpierw wygeneruj ECG
dla testowych okien PPG, zachowując kolejność odpowiadającego im ECG.
Bezpośrednie uruchomienie `eval.py` korzysta z danych syntetycznych z jego
`main()`, a nie z wag wytrenowanych przez `train.py`.

Pozostałe pliki:

| Plik | Zastosowanie |
| --- | --- |
| `code/alyx_data.py` | Odczyt i preprocessing Who is Alyx?. |
| `code/signal_preprocessing.py` | Wspólne funkcje filtracji, resamplingu i segmentacji; loader Alyx używa interpolacji liniowej w swoim kodzie. |
| `code/self_supervised_transformation.py` | Transformacje ECG; samodzielne `main()` jest demonstracją na danych syntetycznych. |
| `code/af_detection.py` | Odrębny eksperyment detekcji AF, z własnymi ustawieniami i przygotowaniem danych. |
| `code/baseline_analysis.py` | Analiza linii bazowej rzeczywistego i wygenerowanego ECG. |
| `code/data.py` | Starszy skrypt z własnymi ścieżkami i dodatkowymi zależnościami; nie jest wymagany przez obecny `train.py`. |

## 10. Testy i rozwiązywanie problemów

### Testy techniczne

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Testy korzystają z małych danych syntetycznych i tymczasowych katalogów.
Sprawdzają m.in. odczyt czasu, indeksowane nagrania, zachowanie przerw,
normalizację, podział uczestników, własny loader oraz krok uczenia SSL/GAN.
Nie wymagają pobranego pełnego zbioru Alyx.

### Typowe problemy

| Objaw | Co sprawdzić lub zmienić |
| --- | --- |
| `Missing dataset directory` | Podaj w `--data-dir` katalog zawierający `players`; sprawdź domyślną lokalizację. |
| `No PPG.json / ECG.json pairs` | Sprawdź strukturę katalogów i pobierz JSON-y, jeśli masz tylko pliki `.dvc`. |
| `No common signal interval` | ECG i PPG danej sesji nie mają wspólnego czasu; sprawdź daty i strefy czasowe. |
| `No usable paired windows` | Sprawdź format czasu, wspólny przedział, przerwy i długość ciągłych fragmentów. Filtr wymaga minimum 1204 próbek; samo skrócenie okna tego nie zmieni. Dla Alyx używaj loadera obsługującego indeksy próbek. |
| `At least three subjects...` | Zapewnij użyteczne pary okien dla co najmniej trzech uczestników. |
| `No usable windows for held-out subject` | Wybierz ID obecny w danych po preprocessingu. |
| Błąd rozmiarów `520` i `512` | Użyj np. `--sampling-rate 128 --window-seconds 4`, aby `T` dzieliło się przez 16. |
| `Window exceeds model positional embedding capacity` | Skróć okno, zmniejsz częstotliwość docelową lub świadomie zmień pojemność embeddingów w konfiguracji. |
| Brak pamięci CUDA | Zmniejsz `--batch-size`, np. do `2`; w razie potrzeby skróć okno, zachowując jego długość w próbkach podzielną przez 16. |
| CUDA niedostępna | Sprawdź `torch.cuda.is_available()`, sterownik i instalację PyTorch; uruchom z `--device cpu`. |
| Brak modułu przy `--data-loader` | Sprawdź zapis `moduł:funkcja` i dostępność modułu, np. jako pliku w `code/`. |
| Brak pliku z wagami po treningu | Zapis nie jest automatyczny; dodaj go zgodnie z sekcją 9. |

Do porównywania eksperymentów zapisuj ustawienia CLI, zmiany w konfiguracji
modelu i transformacji, wersje zależności, ziarno oraz ID uczestników
w podzbiorach. Samo ustawienie ziarna nie gwarantuje identycznych wyników
między różnymi urządzeniami i wersjami bibliotek.
