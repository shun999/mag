--- 
name: commit 
description: Stage and commit current changes 
disable-model-invocation: true 
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) 
--- 

Stage the current changes, generate a commit message that describes the changes, and commit them. 
Use the Conventional Commits format (feat/fix/refactor/docs, etc.).