import matplotlib as mpl
import pandas as pd
import mplcatppuccin
import seaborn as sns
import paths
import logging

# Plot style
TRANSPARENT = True
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
mpl.style.use("mocha")

data = pd.read_csv(paths.data.FILE)
target_label = 'phishing'
y = data[target_label]
X = data.drop(target_label, axis = 1)

fetched_features = [
    'time_response',
    'domain_spf',
     'asn_ip',
     'time_domain_activation',
     'time_domain_expiration',
     'qty_ip_resolved',
     'qty_nameservers',
     'qty_mx_servers',
     'ttl_hostname',
     'tls_ssl_certificate',
     'qty_redirects',
     'url_google_index',
     'domain_google_index'
]

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
