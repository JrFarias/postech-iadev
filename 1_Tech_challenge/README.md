# Tech Challenge

## Como iniciar o jupyterlab ?

```
make docker-start
```

Acesse a URL: [jupyterlab](http://localhost:8888/)


## Como para o jupyterlab ?

```
make docker-stop
```


## Como instalar novas libs ?

```
poetry add [lib-name]
```


### Não tenho o poetry instalado

```
Mac os
 -  brew install pipx

Linux
 - sudo apt update
 - sudo apt install pipx
 - pipx ensurepath

Others: [documentação](https://pipx.pypa.io/stable/installation/)


pipx install poetry
```