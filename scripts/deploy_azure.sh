#!/usr/bin/env bash
# Deploy the marketplace assistant to Azure Container Apps with a STATIC
# outbound IP, so the Atlas allow-list can name one address instead of
# opening the cluster holding real customers to the whole internet.
#
# That static IP is the entire reason this is more than one command.
# Container Apps only has predictable egress when its environment sits in
# a VNet whose subnet routes outbound through a NAT gateway. Steps 2-4
# build that; everything else is ordinary.
#
# Run it a step at a time and read the output. Do not pipe it to bash.
#
#   bash scripts/deploy_azure.sh          # prints the steps, runs nothing
#
set -euo pipefail

# Git Bash on Windows rewrites any argument that looks like a Unix path
# into a Windows one before a native binary sees it, so an Azure resource
# ID arrives as 'C:/Program Files/Git/subscriptions/...' and the API
# rejects it. Harmless-looking and confusing to diagnose, because echo is
# a builtin and prints the ID correctly.
export MSYS_NO_PATHCONV=1

# ── Names. Change RG if you already have one you'd rather use. ────────
RG=marketplace-demo-rg
LOC=southeastasia      # beside the Atlas cluster - see step 0; OpenAI is in eastus
VNET=marketplace-vnet
SUBNET=marketplace-aca-subnet
NATGW=marketplace-nat
NATIP=marketplace-nat-ip
ENVNAME=marketplace-env
ACR=marketplaceacr$RANDOM       # must be globally unique, lowercase, no dashes
APP=commerce-assistant

cat <<'BANNER'
This script is a reference, not an installer. Copy each block, run it,
check the output, then run the next. Steps 2-4 create billable
infrastructure (a NAT gateway is roughly Rs 2,700/month, prorated).
BANNER
exit 0

# ══════════════════════════════════════════════════════════════════════
# 0. Log in, and find where your Azure OpenAI already lives
# ══════════════════════════════════════════════════════════════════════
az login
az account show --output table

# Put the app in the SAME region as the OpenAI resource it calls one to
# three times per answer. If this prints something other than
# centralindia, change LOC at the top of this file to match it.
az cognitiveservices account list \
  --query "[].{name:name, region:location}" --output table

# ══════════════════════════════════════════════════════════════════════
# 1. Resource group
# ══════════════════════════════════════════════════════════════════════
az group create --name "$RG" --location "$LOC" --output table

# ══════════════════════════════════════════════════════════════════════
# 2. Network. The /23 is not padding - Container Apps refuses a
#    smaller infrastructure subnet.
# ══════════════════════════════════════════════════════════════════════
az network vnet create \
  --resource-group "$RG" --name "$VNET" \
  --address-prefix 10.10.0.0/16 \
  --subnet-name "$SUBNET" --subnet-prefix 10.10.0.0/23 \
  --output table

# ══════════════════════════════════════════════════════════════════════
# 3. The static IP, and the NAT gateway that forces egress through it
# ══════════════════════════════════════════════════════════════════════
az network public-ip create \
  --resource-group "$RG" --name "$NATIP" \
  --sku Standard --allocation-method Static --zone 1 2 3 \
  --output table

az network nat gateway create \
  --resource-group "$RG" --name "$NATGW" \
  --public-ip-addresses "$NATIP" --idle-timeout 10 \
  --output table

az network vnet subnet update \
  --resource-group "$RG" --vnet-name "$VNET" --name "$SUBNET" \
  --nat-gateway "$NATGW" \
  --output table

# THIS is the address Atlas must allow. Write it down.
az network public-ip show \
  --resource-group "$RG" --name "$NATIP" \
  --query ipAddress --output tsv

# ══════════════════════════════════════════════════════════════════════
# 4. Container Apps environment, pinned into that subnet
# ══════════════════════════════════════════════════════════════════════
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

SUBNET_ID=$(az network vnet subnet show \
  --resource-group "$RG" --vnet-name "$VNET" --name "$SUBNET" \
  --query id --output tsv)

az containerapp env create \
  --resource-group "$RG" --name "$ENVNAME" --location "$LOC" \
  --infrastructure-subnet-resource-id "$SUBNET_ID" \
  --output table

# ══════════════════════════════════════════════════════════════════════
# 5. Build the image. az acr build builds in Azure, so you do not need
#    Docker running locally and the 270 MB never crosses your uplink.
# ══════════════════════════════════════════════════════════════════════
az acr create --resource-group "$RG" --name "$ACR" \
  --sku Basic --admin-enabled true --output table

# ACR Tasks - which is what `az acr build` uses - is blocked on trial and
# sponsored subscriptions (TasksOperationsNotAllowed). Build locally and
# push instead; same image, but a few hundred MB goes up your uplink.
#
#   az acr build --registry "$ACR" --image commerce-assistant:v1 .
#
IMG="$ACR.azurecr.io/commerce-assistant:v1"
docker build -t "$IMG" .
az acr login -n "$ACR"
docker push "$IMG"

# ══════════════════════════════════════════════════════════════════════
# 6. Deploy. Secrets are set separately from env vars so the values are
#    stored encrypted and never appear in `az containerapp show`.
# ══════════════════════════════════════════════════════════════════════
ACR_PASS=$(az acr credential show --name "$ACR" --query "passwords[0].value" --output tsv)

az containerapp create \
  --resource-group "$RG" --name "$APP" --environment "$ENVNAME" \
  --image "$ACR.azurecr.io/commerce-assistant:v1" \
  --registry-server "$ACR.azurecr.io" \
  --registry-username "$ACR" --registry-password "$ACR_PASS" \
  --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 1 \
  --cpu 1 --memory 2Gi \
  --output table

# --min-replicas 1, not 0: scale-to-zero saves nothing worth having here
# and makes the client's first click wait for a cold start.
# --max-replicas 1 keeps Azure spend bounded on a link anyone you sent it
# to can open.

# Secrets. Replace every <...> with the value from your .env.
az containerapp secret set \
  --resource-group "$RG" --name "$APP" \
  --secrets \
    mongodb-uri="<MONGODB_URI>" \
    jwt-secret="<JWT_SECRET>" \
    azure-key="<AZURE_OPENAI_API_KEY>" \
    redis-url="<REDIS_URL>" \
    demo-code="<invent one, e.g. marketplace-live-2026>"

az containerapp update \
  --resource-group "$RG" --name "$APP" \
  --set-env-vars \
    MONGODB_DATABASE=marketplace \
    DEMO_UI_ENABLED=true \
    WEB_CONCURRENCY=2 \
    AZURE_OPENAI_ENDPOINT="<AZURE_OPENAI_ENDPOINT>" \
    AZURE_OPENAI_DEPLOYMENT="<AZURE_OPENAI_DEPLOYMENT>" \
    AZURE_EMBEDDING_DEPLOYMENT="<AZURE_EMBEDDING_DEPLOYMENT>" \
    MONGODB_URI=secretref:mongodb-uri \
    JWT_SECRET=secretref:jwt-secret \
    AZURE_OPENAI_API_KEY=secretref:azure-key \
    REDIS_URL=secretref:redis-url \
    DEMO_ACCESS_CODE=secretref:demo-code

# ══════════════════════════════════════════════════════════════════════
# 7. Atlas. Do this BEFORE testing, or every request fails on connect.
# ══════════════════════════════════════════════════════════════════════
#   Atlas -> Network Access -> ADD IP ADDRESS
#   Paste the address from step 3, comment "Azure Container Apps NAT".
#   Then DELETE any 0.0.0.0/0 entry - leaving it makes the whole
#   NAT gateway pointless.

# ══════════════════════════════════════════════════════════════════════
# 8. Verify
# ══════════════════════════════════════════════════════════════════════
URL=https://$(az containerapp show --resource-group "$RG" --name "$APP" \
  --query properties.configuration.ingress.fqdn --output tsv)
echo "$URL/demo"

curl -sS "$URL/health"                                   # expect 200
curl -sS -o /dev/null -w '%{http_code}\n' "$URL/demo/accounts"          # expect 401
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "X-Demo-Code: <your code>" "$URL/demo/accounts"                    # expect 200

az containerapp logs show --resource-group "$RG" --name "$APP" --tail 50

# ══════════════════════════════════════════════════════════════════════
# 9. Tearing it down. The credit eventually expires and the NAT gateway
#    bills until deleted, whether or not anyone opens the link.
# ══════════════════════════════════════════════════════════════════════
# az group delete --name "$RG" --yes --no-wait
