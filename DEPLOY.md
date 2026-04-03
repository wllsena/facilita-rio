# Deploy

## Fluxo

```
Push no main
    |
    v
GitHub Actions (.github/workflows/deploy.yml)
    |
    +-- 1. Lint (ruff check)
    +-- 2. Testes (pytest, 73 testes)
    |
    v  (só se 1 e 2 passaram)
    |
    +-- 3. SSH na EC2
    +-- 4. git pull
    +-- 5. docker compose build (container antigo ainda rodando)
    +-- 6. docker compose up -d (swap)
    +-- 7. Health check (até 5 min)
    +-- 8. Limpeza de imagens antigas
```

Pull requests na `main` rodam apenas lint + testes (sem deploy).

## Downtime

O build da imagem Docker roda **com o container antigo ainda servindo requests**. O downtime acontece apenas durante o startup do novo container (~50s na máquina atual, por causa do carregamento dos modelos ML). Não é zero-downtime, mas a janela é mínima.

Se o health check falhar após 5 minutos, o deploy falha e aparece erro no GitHub Actions.

## Infraestrutura

| Componente | Detalhe |
|---|---|
| Servidor | EC2 `t4g.micro` (ARM64, 1.8GB RAM, 2GB swap) |
| Domínio | facilita-rio.com |
| HTTPS | Let's Encrypt via certbot (renovação automática) |
| Proxy reverso | nginx (80/443 para localhost:8000) |
| Aplicação | Docker container rodando como `appuser` (non-root) |
| CI/CD | GitHub Actions |

## Configuração

### Secrets do GitHub Actions

Em Settings > Secrets and variables > Actions:

| Secret | Descrição |
|---|---|
| `EC2_SSH_KEY` | Chave PEM para SSH na EC2 |
| `EC2_HOST` | Hostname da EC2 |
| `EC2_USER` | Usuário SSH (`ubuntu`) |

### Deploy key

O servidor EC2 tem uma deploy key SSH (`~/.ssh/deploy_key`) registrada no repositório para fazer `git pull` sem autenticação interativa. Configurada em `~/.ssh/config`.

### Arquivos no servidor

| Arquivo | Função |
|---|---|
| `~/deploy.sh` | Script de deploy (git pull, build, swap, health check) |
| `~/facilita-rio/` | Clone do repositório |
| `/etc/nginx/sites-enabled/facilita-rio.com` | Configuração nginx + SSL |
| `/etc/letsencrypt/live/facilita-rio.com/` | Certificados SSL |

## Comandos úteis

```bash
# Health check
curl https://facilita-rio.com/health

# Status do último deploy
gh run list --repo wllsena/facilita-rio- --limit 1

# Logs do deploy
gh run view <run-id> --repo wllsena/facilita-rio- --log

# Logs do container
ssh -i <key> ubuntu@<host> "sudo docker compose -f ~/facilita-rio/docker-compose.yml logs --tail 30"

# Deploy manual (sem push)
ssh -i <key> ubuntu@<host> "bash ~/deploy.sh"

# Reiniciar sem rebuild
ssh -i <key> ubuntu@<host> "cd ~/facilita-rio && sudo docker compose restart"
```
