@echo off
echo ============================================================
echo    INSTALLATION RAPIDE RTM TRADING
echo ============================================================
echo.

REM Créer le dossier models
if not exist "models" (
    echo Création du dossier models...
    mkdir models
    echo [OK] Dossier models créé
) else (
    echo [INFO] Dossier models existe déjà
)

echo.
echo ============================================================
echo CREATION DES FICHIERS PYTHON
echo ============================================================

REM Créer __init__.py
echo # Models package > models\__init__.py
echo [OK] models\__init__.py créé

REM Instructions pour la suite
echo.
echo ============================================================
echo PROCHAINES ETAPES
echo ============================================================
echo.
echo Maintenant, créez ces fichiers dans le dossier models\ :
echo.
echo 1. models\common.py
echo 2. models\ema.py
echo 3. models\layers.py
echo 4. models\losses.py
echo 5. models\sparse_embedding.py
echo.
echo Je vais vous afficher le contenu de chaque fichier...
echo.
pause

echo.
echo ============================================================
echo Ouvrez le Bloc-notes pour chaque fichier et copiez-collez :
echo ============================================================
echo.
echo Appuyez sur une touche pour continuer...
pause > nul

REM Ouvrir le bloc-notes pour chaque fichier
start notepad models\common.py
start notepad models\ema.py
start notepad models\layers.py
start notepad models\losses.py
start notepad models\sparse_embedding.py

echo.
echo [INFO] 5 fenêtres Notepad ouvertes
echo [INFO] Copiez le contenu depuis les instructions ci-dessous
echo.
echo ============================================================
echo INSTALLATION TERMINEE
echo ============================================================
pause
