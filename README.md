# PJT_BOK_1
텍스트 마이닝을 활용한 시장지표 예측 모델링

---

# 간단한 github 명령어 (순서대로 해야 됨)
- git push origin --delete 브랜치명 (원격저장소(github)에서 브랜치를 삭제) 
- git fetch origin --prune (원격저장소에서 이미 삭제된 브랜치들을 내 로컬저장소에서도 깔끔하게 정리)
- git branch -d 브랜치명 (내가 만든 브랜치를 로컬저장소에서 삭제)

---

# 병합 시 할 일
- git commit -m "메시지" (병합하기 전에 작업한 브랜치에서 commit)
- git checkout develop (병합할 브랜치로 이동)
- git pull origin develop (최신 내용 가져오기)
- git merge 브랜치명 (합칠 브랜치 명 입력해서 합치기)
- git push origin develop (원격 저장소에 푸시)
- git branch --merged (이미 병합이 완료되어 삭제해도 안전한 브랜치 목록을 보여줌, 보고 브랜치 삭제)