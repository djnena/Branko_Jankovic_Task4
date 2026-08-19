# Branko_Jankovic_Task4

Sta projekat radi?

Ovaj projekat se bavi analizom podataka o automobilima i predvidjanjem njihove cene.

U okviru projekta:

- ucitavamo i istrazujemo podatke o automobilima;
- proveravamo tipove podataka, nedostajuce i ekstremne vrednosti;
- cistimo i standardizujemo podatke;
- kreiramo nove karakteristike, kao sto su car_age, mileage_per_year i engine_volume_liters;
- pripremamo podatke za masinsko ucenje;
- testiramo vise regresionih algoritama;
- uporedjujemo njihove rezultate pomocu MAE, MSE, RMSE i R² metrika;
- biramo najbolji model za predvidjanje cene automobila.

Finalni cilj projekta je da na osnovu karakteristika automobila, kao sto su godina proizvodnje, kilometraza, zapremina motora, proizvodjac, model, gorivo, menjac i druge karakteristike, sto preciznije predvidimo njegovu cenu (priceUSD).

Kao najbolji model u nasoj analizi izabran je Random Forest Regressor, sa R² rezultatom od 0,8943.

Koji skup podataka koristi?

Skup podataka sadrzi 56.244 redova i podatke o karakteristikama kao sto su:

make – proizvodjac automobila;
model – model automobila;
priceUSD – cena automobila u americkim dolarima (ciljna promenljiva);
year – godina proizvodnje;
condition – stanje automobila;
mileage(kilometers) – kilometraza;
fuel_type – vrsta goriva;
volume(cm3) – zapremina motora;
color – boja;
transmission – tip menjaca;
drive_unit – pogon;
segment – segment vozila.

Tokom pripreme podataka izvedene su i nove karakteristike, kao sto su car_age, mileage_per_year i engine_volume_liters.

Ciljna promenljiva je priceUSD.

Kako se pokrece kod?

Analiza podataka se pokrece kroz Jupyter Notebook-u. Nakon analize pustamo fajlove .py, sledecim redom:
- data_cleaning.py - ciscenje podataka
- feature_engineering.py - dodavanje novih karakteristika
- data_preprocessing.py - predprocesiranje
- model_training.py - treniranje modela
- model_evaluation.py - evaluacija modela
- model_comparison.py - poredjenje tri regresiona modela
- model_final.py - kreiranje finalnog modela

Koje modele smo testirali?

U projektu smo testirali cetiri regresiona modela za predvidjanje cene automobila:

- Linear Regression – osnovni linearni model koji predstavlja pocetnu referentnu tacku
- Decision Tree Regressor – model zasnovan na stablima odlucivanja koji moze da prepozna nelinearne odnose
- Random Forest Regressor – koristi veci broj stabala i njihovim kombinovanjem daje stabilnija i preciznija predvidjanja
- Gradient Boosting Regressor – gradi modele sukcesivno, pri cemu svaki novi model pokusava da popravi greske prethodnog

Koji rezultat smo dobili?

Nakon testiranja cetiri regresiona modela, najbolji rezultat ostvario je Random Forest Regressor.

Dobijene metrike za finalni model su:

MAE: 1.051,43 USD
MSE: 6.927.934
RMSE: 2.632,10 USD
R²: 0,8943

To znaci da Random Forest u proseku gresi oko 1.051 USD u predvidjanju cene i uspeva da objasni priblizno 89,43% varijacije cena automobila.

Koji model smo izabrali i zasto?

Kao finalni model izabrali smo Random Forest Regressor.

Izabran je zato sto je medju testiranim modelima ostvario najbolje rezultate prema svim glavnim regresionim metrikama:

MAE = 1.051,43 USD → najmanja prosecna greska
RMSE = 2.632,10 USD → najmanja greska koja posebno uzima u obzir velike greske
R² = 0,8943 → najveca vrednost, odnosno model objasnjava oko 89,43% varijacije cena

U poredjenju sa Linear Regression modelom, Random Forest je znacajno precizniji: MAE je smanjen sa 2.089,02 USD na 1.051,43 USD, a R² povecan sa 0,7023 na 0,8943.
