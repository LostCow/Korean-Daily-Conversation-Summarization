# Korean Daily Conversation Summarization

[인공지능(AI)말평 일상대화요약(가 유형)](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=146&clCd=END_TASK&subMenuId=sub01) 제출 코드

## 실행 방법

### 환경 설정
```bash
$ pip install -r requirements.txt
```

### 데이터 다운로드
```bash
https://kli.korean.go.kr/taskOrdtm/taskDownload.do?taskOrdtmId=146&clCd=END_TASK&subMenuId=sub02
```

## 모델 학습
```bash
$ sh scripts/run_train_tapt.sh
$ sh scripts/run_train_sft.sh
```

## 모델 추론
```bash
$ sh scripts/run_infer.sh
```

## License
[CC-BY-NC-4.0](https://choosealicense.com/licenses/cc-by-4.0/)