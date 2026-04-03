# Deploy

## Como funciona

Cada push na branch `main` do GitHub dispara um deploy automático na EC2 via GitHub Actions.

```
Push no main → GitHub Actions → SSH na EC2 → git pull → docker build → swap container → health check
```

### Fluxo detalhado

1. **GitHub Actions** detecta push no `main` e executa `.github/workflows/deploy.yml`
2. O workflow faz SSH na EC2 usando a chave armazenada nos secrets
3. Executa `~/deploy.sh` no servidor, que faz:
   - `git pull origin main` — baixa o código novo
   - `docker compose build` — builda a nova imagem **sem derrubar o container atual**
   - `docker compose up -d` — troca pro novo container (downtime apenas no startup, ~50s)
   - Aguarda até 5 minutos pelo health check (`/health`)
   - Limpa imagens Docker antigas
   - Se o health check falhar, o deploy falha e aparece erro no GitHub Actions

### Downtime

O build roda com o container antigo ainda servindo requests. O downtime acontece apenas durante o startup do novo container (~50s na máquina atual). Não é zero-downtime completo, mas minimiza a janela.

## Infraestrutura

| Componente | Detalhe |
|---|---|
| Servidor | EC2 `t4g.micro` (ARM64, 1.8GB RAM, 2GB swap) |
| IP | 34.197.46.213 |
| Domínio | facilita-rio.com |
| HTTPS | Let's Encrypt (renovação automática via certbot) |
| Proxy reverso | nginx (porta 80/443 → localhost:8000) |
| Aplicação | Docker container (uvicorn na porta 8000) |

## Secrets do GitHub Actions

Configurados em Settings → Secrets and variables → Actions:

| Secret | Descrição |
|---|---|
| `EC2_SSH_KEY` | Chave PEM para SSH na EC2 |
| `EC2_HOST` | Hostname da EC2 |
| `EC2_USER` | Usuário SSH (`ubuntu`) |

## Deploy key

O servidor EC2 tem uma deploy key SSH (`~/.ssh/deploy_key`) registrada no repositório GitHub para fazer `git pull` sem autenticação interativa.

## Arquivos relevantes

| Arquivo | Onde | Função |
|---|---|---|
| `.github/workflows/deploy.yml` | Repositório | Workflow do GitHub Actions |
| `~/deploy.sh` | EC2 | Script de deploy executado pelo workflow |
| `/etc/nginx/sites-enabled/facilita-rio.com` | EC2 | Configuração nginx + SSL |
| `/etc/letsencrypt/live/facilita-rio.com/` | EC2 | Certificados SSL |

## Deploy manual

Se precisar fazer deploy sem push:

```bash
ssh -i ~/Documents/money-miles/djangoletsencrypt.pem ubuntu@ec2-34-197-46-213.compute-1.amazonaws.com "bash ~/deploy.sh"
```

## Verificação

```bash
# Health check
curl https://facilita-rio.com/health

# Logs do container
ssh -i <key> ubuntu@<host> "sudo docker compose -f ~/facilita-rio/docker-compose.yml logs --tail 20"

# Status do último deploy
gh run list --repo wllsena/facilita-rio- --limit 1
```
