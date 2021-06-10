

# CS475 Final Project (AttacKing Sejong)

### Replication BERT-ATTACK (EMNLP 2019) in Korean [[PAPER]](https://arxiv.org/abs/2004.09984)
- The reason why our project name is 'AttacKing Sejong' 😂 
- Model : KLUE-bert-base
- Dataset : KLUE-TC (a.k.a YNAT) (+ Maybe NSMC, KLUE-NLI)

<br>


## Convention commits
[Conventional Commits/Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type)

- build: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- ci: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- docs: Documentation only changes
- feat: A new feature
- fix: A bug fix
- perf: A code change that improves performance
- refactor: A code change that neither fixes a bug nor adds a feature
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- test: Adding missing tests or correcting existing tests

## Dependencies
- Python 3.8.5
- PyTorch 1.7.1
- transformers 4.6.1

## References
- [BERT-Attack](https://github.com/LinyangLee/BERT-Attack)
- [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
- [KLUE-TC (a.k.a YNAT)](https://klue-benchmark.com/tasks/66/overview/description)