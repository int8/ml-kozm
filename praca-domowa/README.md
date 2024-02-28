## praca domowa: instrukcje



1. uruchom VM 
2. wejdź do terminala 
3. `cd` # == idź do katalogu domowego 
4. `cd ml-kozm` # wejdź do repo 
5. `mkdir -p local_work` # utwórz katalog w którym bedziesz miał kopie swojej pracy z poprzednich zajęć
6. `cp *.ipynb local_work` # skopiuj swoją pracę do local_work (ten katalog jest ignorowany przez git)
7. `ls local_work` # upewnij się że to polecenie listuje Twoje notebooki 
8. `git stash` 
9. `git branch --set-upstream-to=origin/master`
10. `git pull`
11. `git checkout homework`



W tym momencie o ile wszystko poszło pomyślnie, 
powinieneś/powinnaś zobaczyć nowy katalog `praca-domowa`


Aktywuj wirtualne środowisko python 

```shell
source mlcourse/bin/activate
```
oraz uruchom `jupyter notebook`

juz w jupyterze nawiguj do katalogu `praca-domowa` 

1. Uruchom i rozwiąż Task1A
2. Uruchom i rozwiąż Task1A
3. Uruchom i rozwiąż Task2

