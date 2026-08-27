# SparkDeck QuickStart: two DGX Sparks over Tailscale

This guide starts with two unpaired NVIDIA DGX Spark systems and ends with one SparkDeck cluster that can be managed from either machine. It uses direct private Tailscale IPv4 URLs because they are the shortest path to a working cluster. An optional Tailscale Serve HTTPS setup is included afterward.

> [!IMPORTANT]
> SparkDeck is a privileged local management application. Do not port-forward TCP 7878, place it behind Tailscale Funnel, or expose it directly to the public internet. Keep both systems in the same private tailnet and control access with Tailscale grants or ACLs.

## What you will build

| Machine | SparkDeck role | Example Tailscale IP | Private SparkDeck URL |
| --- | --- | --- | --- |
| Spark 1 | Controller | `100.101.10.11` | `http://100.101.10.11:7878` |
| Spark 2 | Joined node | `100.101.10.12` | `http://100.101.10.12:7878` |

Use the real IP printed on each machine. Do not copy the example addresses above.

## Before you begin

On both Sparks, confirm:

```bash
python3 --version
node --version
npm --version
docker info
nvidia-smi
```

SparkDeck requires Python 3.11 or newer. Its locked Vite toolchain requires Node.js `^20.19.0` or `>=22.12.0`; earlier Node 20 releases are not supported. Linux with Docker and the NVIDIA Container Toolkit is the recommended GPU-worker environment. You also need Git and permission to run Docker.

## 1. Install Tailscale on both Sparks

Run the official one-line Linux installer on **Spark 1** and **Spark 2**:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

Then connect each machine:

```bash
sudo tailscale up
```

Open the authentication URL printed in the terminal. Sign both machines into the **same tailnet**. Tailscale documents this flow in [Install Tailscale on Linux](https://tailscale.com/docs/install/linux).

## 2. Get the correct IP from each Spark

On Spark 1:

```bash
tailscale ip -4
```

Write the returned `100.x.y.z` address down as `SPARK_1_IP`.

On Spark 2:

```bash
tailscale ip -4
```

Write that different address down as `SPARK_2_IP`. Tailscale assigns each machine its own stable address; the single name shown on the Tailscale DNS page is the tailnet's DNS suffix, not a device URL. Device-specific names are listed on the Tailscale **Machines** page and normally look like `spark-1.<tailnet-name>.ts.net`.

Confirm both peers are visible:

```bash
tailscale status
```

From Spark 2, test Spark 1:

```bash
tailscale ping <SPARK_1_IP>
```

Replace the placeholder, including the angle brackets, with Spark 1's real address.

## 3. Install SparkDeck on both Sparks

Run the following on each machine:

```bash
git clone https://github.com/hyudryu/SparkDeck.git
cd SparkDeck
./run.sh
```

The first run creates a Python virtual environment, installs dependencies, builds the frontend when needed, and starts SparkDeck on TCP port 7878. Keep this terminal open for the initial setup.

Open the local UI on each machine to verify it started:

```text
http://localhost:7878
```

If UFW is active and a peer cannot connect, allow only the Tailscale interface:

```bash
sudo ufw allow in on tailscale0 to any port 7878 proto tcp
```

Do not create an unrestricted public firewall rule and do not configure router port forwarding.

## 4. Verify the private URLs

From Spark 2, open Spark 1's URL in a browser:

```text
http://<SPARK_1_IP>:7878
```

From Spark 1, open Spark 2's URL:

```text
http://<SPARK_2_IP>:7878
```

Direct HTTP is still encrypted between Tailscale peers by Tailscale's WireGuard transport, but the browser treats it as an insecure origin. Use the optional HTTPS section below if a browser feature requires a secure origin.

## 5. Make Spark 1 the controller

1. Open `http://localhost:7878` on Spark 1.
2. Select **Cluster** in the left navigation.
3. Change the current node's name to something recognizable, such as `Spark 1`.
4. In **Add a node from this controller**, copy Spark 1's private access URL and the current one-time pairing code.

![Cluster onboarding with illustrative dummy data](docs/screenshots/readme/cluster-management-dark.png)

The pairing code is temporary. Generate or copy a fresh code immediately before joining Spark 2.

## 6. Join Spark 2

1. Open `http://localhost:7878` on Spark 2. Do this locally so you are editing Spark 2's onboarding state.
2. Select **Cluster**.
3. Choose **Join this node to another controller**.
4. For **Existing cluster entry URL**, enter `http://<SPARK_1_IP>:7878`.
5. Enter the pairing code shown on Spark 1.
6. Set the node name to `Spark 2`.
7. Confirm Spark 2's advertised URL is `http://<SPARK_2_IP>:7878`.
8. Submit the form.

Only Spark 2 joins. If Spark 2 previously controlled other nodes, each of those machines must leave its old cluster and join Spark 1 separately.

## 7. Verify two-way access

Open either private URL:

```text
http://<SPARK_1_IP>:7878
http://<SPARK_2_IP>:7878
```

Both should show the same controller-owned cluster inventory with `Spark 1` and `Spark 2` online. Spark 2 is an alternate private entry point: management requests are forwarded to Spark 1. This is two-way visibility, but not automatic failover. If Spark 1 is offline, running models can continue but cluster management is unavailable.

On **Dashboard**, verify both node cards are present. On **Cluster**, verify both nodes are online and named correctly.

## 8. Add Hugging Face access

If you use gated or private repositories:

1. Create a read token in Hugging Face.
2. Open **Settings → Hugging Face access**.
3. Save the token once for the cluster.

SparkDeck reports whether a token exists but never returns the stored value to the browser.

## 9. Download and deploy a first model

1. Open **Explore** and search Hugging Face.
2. Expand a model row and inspect the fit indicator.
3. Choose **Deploy** or open **Models** to create a saved configuration.
4. Select the node that should download and run the model.
5. For TP2, select exactly two eligible nodes. Nodes without the weights remain disabled until the model is downloaded or transferred there.
6. When the deployment is ready, open **Chat** and select it.

Use **Storage** to copy an existing complete Hugging Face cache entry from one Spark to another instead of downloading it again. Virtual NAS must be enabled before transfers can be queued.

## Optional: use Tailscale Serve HTTPS

Run this on each Spark after SparkDeck is listening locally:

```bash
sudo tailscale serve --bg --https=443 localhost:7878
```

The command prints that machine's URL, normally:

```text
https://<machine-name>.<tailnet-name>.ts.net
```

Each machine has a different machine name even though the Tailscale DNS page shows one shared tailnet suffix. If you already paired the nodes with the HTTP URLs in step 6, switch the stored cluster URLs by opening Cluster on Spark 2, choosing **Leave cluster**, generating a fresh pairing code on Spark 1, and joining again. During that rejoin, use Spark 1's printed HTTPS URL as **Existing cluster entry URL** and Spark 2's printed HTTPS URL as **This node's advertised Tailscale URL**. Verify the proxy with:

```bash
tailscale serve status
```

[Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve) is private to the tailnet. **Tailscale Funnel is public and must not be used for SparkDeck.** HTTPS certificate names are recorded in public certificate-transparency logs, so rename machines first if their names are sensitive.

## Keep SparkDeck running after logout

The repository includes a user-service template that assumes the checkout is at `~/SparkDeck`. First, return to the terminal where `./run.sh` is still running and press **Ctrl+C**. Do this on each Spark before starting the service so two processes do not compete for port 7878.

Then enable lingering for the signed-in Linux user so its systemd user manager—and SparkDeck—can continue after logout:

```bash
sudo loginctl enable-linger "$USER"
```

Install and start the user service:

```bash
mkdir -p ~/.config/systemd/user
cp sparkdeck.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sparkdeck.service
```

Check it with:

```bash
systemctl --user status sparkdeck.service
journalctl --user -u sparkdeck.service -f
```

If the checkout is elsewhere, edit the copied unit before enabling it. Never put tokens directly in the committed service file.

## Troubleshooting

### Spark 2 cannot open Spark 1

1. Run `tailscale status` on both machines.
2. Run `tailscale ping <SPARK_1_IP>` from Spark 2.
3. Confirm SparkDeck is listening with `curl http://localhost:7878/api/v1/onboarding` on Spark 1.
4. Check `sudo ufw status` and allow port 7878 only on `tailscale0` if needed.
5. Confirm both machines are in the same tailnet and allowed by its grants or ACLs.

### The pairing code fails

Return to Cluster on Spark 1 and use the newest displayed code. Pairing codes are one-time and expire. Confirm the controller URL belongs to Spark 1 and the advertised URL belongs to Spark 2.

### I see only one tailnet DNS name

That is correct. The DNS page shows the tailnet suffix. Each device has its own machine name and full MagicDNS name on the Machines page. `tailscale ip -4` is the simplest way to get the unambiguous address needed for the direct HTTP setup.

### A machine from an old cluster did not follow

Membership is per machine. On that machine's own local SparkDeck page, use **Leave cluster**, then join it directly to Spark 1 with a fresh pairing code. Joining a former controller never imports its child-node registry.

### The controller is offline

Joined nodes are alternate entry points, not automatic controller failover. Restore Spark 1 for cluster management. Local onboarding remains available on each machine for recovery.

## Next steps

- Read the [product manual](docs/PRODUCT_MANUAL.md).
- Configure **Settings → Hugging Face access** for gated models.
- Review **Benchmarks → Community sharing** before opting in.
- Enable **Storage → Virtual NAS** if you want cluster weight transfers.
- Use Tailscale grants or ACLs to limit which users and devices can reach TCP 7878.
