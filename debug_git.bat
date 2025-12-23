@echo off
echo Running git init... > git_log.txt
git init >> git_log.txt 2>&1
echo Running git status... >> git_log.txt
git status >> git_log.txt 2>&1
echo Adding files... >> git_log.txt
git add . >> git_log.txt 2>&1
echo Committing... >> git_log.txt
git commit -m "Upload complete suite" >> git_log.txt 2>&1
echo Setting main branch... >> git_log.txt
git branch -M main >> git_log.txt 2>&1
echo Adding remote... >> git_log.txt
git remote add origin https://github.com/SaChIn5419/Sector_Rotation_python.git >> git_log.txt 2>&1
echo Pushing... >> git_log.txt
git push -f -u origin main >> git_log.txt 2>&1
echo Finished! >> git_log.txt
