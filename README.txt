Żeby wytrenować sieci dla wybranych danych odpowiednio zmodyfikować i odpalić plik train.py (wyśle Panu wagi wytrenowanego juz modelu, żeby nie musiał Pan tracic na to czasu).

Żeby odpalić własciwy gaze_tracking , uruchomic plik tests.py, na razie na 1 datasecie który od Pana dostałem(nie trenowałem jeszcze sieci na pozostałych).
W pliku gaze_tracker.py jest cały proces zakodowany, specjalne sekcje są do wizualizacji, proszę je odkomentować jeżeli chce Pan zobaczyc jak wyglada przewidziana elipsa, 
wektory lub środek oka.

Algorytm modelowania oka -> eye_modeling
Starałem się opisać co się dzieje w komentarzach i tutaj poniżej, funkcje są nazwane według papieru świrskiego, i są wykonywane równiez w takiej kolejnosći jak w papierze.
pierwszy krok i najbardziej skomplikowany to unprojection(nwm jak po polsku) wektorów -> unprojection.py


Z tego co nad tym siedze już długi czas(i widac na wizualizacji), to wektory z unprojection po prostu nie pokazują poprawie w strone środka, często są kompletnie w poprzek,
przez co estymowany środek też wychodzi bezsensowny.
Sprawdzałem sam algorytm unprojection mnóstwo razy, nawet sprawdziłem juz w rozpaczy, kopiując z DeepVoga ich algorytm unprojection i wychodziło to samo.
Skoro elipsa jest przeiwidywana w porządku(widać na wizualizacji), to chyba musi być jakiś durny błąd w stylu zamienione współrzedne albo coś z perspektywą kamery,
ja podchodzę do tego tak że środek obrazu to w perspektywie kamery (0,0) i tak też wszytkie obliczenia są prowadzone. Ale może to coś innego czego ja nie widze, 
może pańskie świeże spojrzenie coś wykryje ;p


=======================================================================================================
ALGORYTM MODELOWANIA OKA

1. TWO CIRCLE UNPROJECTION

uprojection of an ellipse -> find a circle whose projection is a given ellipse.
przestrzen wszystkich możliwych projekcji koła źrenicy można przybliżyć jako stożek z kołem źrenicy w podstawie i ogniskową kamery w wierzchołku.
Wtedy nasza elipsa to przecięcie stożka z powierzchnią obrazu

To oznacza, że możemy zrekonstruować stożek z podstawą jako elipsa źrenicy, wtedy kołowe przecięcie tego stożka to będzie nasze koło elipsy (przeciecie znajdujemy metoda safaee-rad)
tą metodą otrzymujemy pozyjce źrenicy, wektor normalny 
natomiast nie znajac rozmiaru źrenicy nie jestesmy w stanie jednoznacznie okreslic jak daleko te koło jest od kamery 
Chwilowo zakładamy stały promień źrenicy(np. 2 mm).
Po tym powstają nam 2 rozwiązania o takich samych rozmiarach, symetryczne wzdłuż głownej osi ellipsy, z rozwiązania równania kwadratowego (p+, n+ r) , (p-, n-, r)
Na ten moment nie jesteśmy w stanie powiedzieć które jest poprawne.

2. MODELOWANIE

Modelujemy źrenice jako dysk styczny do sferycznej gałki ocznej, 
wtedy wektor wzorku to wektor normalny dysku

3. SPHERE CENTRE ESTIMATE

Mamy koło źrenicy dla każdego obrazka, chcemy teraz znaleźć taką sferę, która jest styczna do każdego koła
Nie mozemy tego znaleźć  w 3D ponieważ nie znam rzeczywistego promienia koła źrenicy i nie wiemy które z 2 rozwiązań jest dobre
Więc przeanalizujemy to w 2D przestrzeni obrazu.
2 wektory normalne z naszych 2 rozwiązań są w 2D równoległe więc możemy wziąc którykolowiek z nich.
Środek naszej gałki ocznej to będzie przęciecie wektorów normalnych naszych dysków (źrenic)
Linie te ze wzgledu na błędy i założenia nie będą się przecinac w jednym punkcie, wiec wybieramy najblższy wszystkim liniom
Dzieki temu mamy wspolrzedne x,y środka źrenicy c
Nie znamy współrzednej z wiec na razie ją fixujemy 

4. SPHERE RADIUS ESTIMATE

każdy wektor normalny źrenicy musi wskazywac kierunek od środka sfery n* (c-p) > 0, 
teraz możemy odrzucić jedno rozwiązania, którego wektor normalny wskazuje do środka sfery

można policzyc środek dysku stycznego do sfery , 
jako przecięcie lini (camera_vertex, p) i lini ze środka sfery do p (c, p)
liczymy punkt najbliższy obu liniom
R - promień sfery to średni dystans od środka sfery do środka źrenicy  mean(R = p-c)

5. CONSISTENT PUPIL ESTIMATE

teraz chcemy policzyć nowe koło źrenicy (p', n' ,r'), styczne do sfery oka 
gdzie 
p' = sp 
p'=c +Rn' 
r'/z' = r/z 
s znajdujemy jako s*p to przeciecie sfery z linią, potem wyliczamy n' i r'
Po tym wszystkim mamy jako taki model ruchy źenicy
