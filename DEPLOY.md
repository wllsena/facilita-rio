# Deploy

## Como Funciona

Quando você envia para o branch `main`:

1. **GitHub Actions** executa lint (`ruff check .`) e testes (`pytest`)
2. Se passarem, faz SSH no servidor de produção
3. Puxa o código mais recente, constrói uma nova imagem Docker (o container antigo continua servindo durante o build)
4. Troca para o novo container
5. Executa health check — se falhar após 5 minutos, o deploy falha

```
Push para main → Lint + Testes → SSH no servidor → Build da imagem → Troca de container → Health check
```

Pull requests executam apenas lint + testes (sem deploy).

**Indisponibilidade:** ~50 segundos enquanto os modelos de ML carregam durante a troca de container. O container antigo serve tráfego durante o build do Docker.

## Infraestrutura

| Componente | Detalhe |
|-----------|---------|
| Servidor | EC2 `t4g.micro` (ARM64, 1.8GB RAM, 2GB swap) |
| Domínio | facilita-rio.com |
| HTTPS | Let's Encrypt via certbot (renovação automática) |
| Proxy reverso | nginx (portas 80/443 → localhost:8000) |
| Aplicação | Container Docker, usuário não-root |
| CI/CD | GitHub Actions (`.github/workflows/deploy.yml`) |

## Configuração

### Secrets do GitHub Actions

Em Settings > Secrets and variables > Actions do repositório, configure:

| Secret | O que é |
|--------|---------|
| `EC2_SSH_KEY` | Chave PEM para acesso SSH ao servidor |
| `EC2_HOST` | Hostname ou IP do servidor |
| `EC2_USER` | Usuário SSH (geralmente `ubuntu`) |

### Arquivos no Servidor

| Caminho | Propósito |
|---------|-----------|
| `~/deploy.sh` | Script de deploy (pull, build, troca, health check) |
| `~/facilita-rio/` | Clone do repositório |
| `/etc/nginx/sites-enabled/facilita-rio.com` | Configuração nginx + SSL |
| `/etc/letsencrypt/live/facilita-rio.com/` | Certificados SSL |

O servidor tem uma deploy key (`~/.ssh/deploy_key`) para git pull sem autenticação interativa.

## Comandos Úteis

```bash
# Verificar se a aplicação está rodando
curl https://facilita-rio.com/health

# Status do último deploy
gh run list --repo wllsena/facilita-rio --limit 1

# Logs do deploy
gh run view <run-id> --repo wllsena/facilita-rio --log

# Logs do container
ssh -i <chave> ubuntu@<host> "sudo docker compose -f ~/facilita-rio/docker-compose.yml logs --tail 30"

# Deploy manual (sem push)
ssh -i <chave> ubuntu@<host> "bash ~/deploy.sh"

# Reiniciar sem rebuild
ssh -i <chave> ubuntu@<host> "cd ~/facilita-rio && sudo docker compose restart"
```
