#Použité metody
##FastText a SVM

Vektorová reprezentace ze vstupních dat byla vytvořena s použitím předtrénovaných vektorů pro češtinu, trénovaných na Common Crawl a Wikipedia textech pomocí fastText modelu, využívajícího CBOW s dimenzí 300, n-gramy délky 5, velikostí okna 5 a minus 10.

Vektory dale zpracovává SVM klasifikátor s radiální bázovou funkcí. Pro každý vstup (text) do modelu jeou jeho vektory zprůměrovány a standardizovány odečtením průměru a podělením standartní odchylkou.

##LSTM

Dále byla pro klasifikaci trénována LSTM síť, která umožňuje naučení delších závislostí ze vstupního textu. Síť tvoří embedding vrstva pro převod vstupu na vektorovou reprezentaci, LSTM vrstvou se 100 neurony, a plně propojenou vrstvou se softmax aktivační funkcí pro klasifikaci. 

#Výsledky a závěry
Vstupní data byla rozdělena na 80% pro trénování a 20% pro testování. Výsledek klasifikace pomocí fastText vektorové reprezentace a SVM byl překvapivě dobrý, přesnost na testovací sadě byla kolem 95%. Při použití LSTM sítě s embedding vrstvou byla výsledná přesnost klasifikace byla přibližně 98%.

Zastoupení jednotlivých kategorií se v datasetu velmi liší, což u jejich klasifikace málo zastoupených skupin činí problémy. Například kategorie hokejbal je ve vstupních datech natolik málo zastoupena, že by mohlo připadat v úvahu ji řešit zvlášť mimo klasifikační model. 

Kvůli problémové klasifikaci u málo zastoupených kategorií byla LSTM síť také trénována s nastavením vah pro jednotlivé kategorie poměrově k jejich četnosti vzorků. Nepoměr mezi kategoriemi je však vysoký a výsledná přesnost u testovacích dat byla nižší (93%).


