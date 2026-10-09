# Build only from the hashed local context; no network at build time.
FROM public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:de98dc02a68765df5688d301134debbd677ccdc4fbd6873af34fafafcc3462ec
USER root
COPY wheelhouse /opt/trainium-artifacts/wheelhouse
COPY debs /opt/trainium-artifacts/debs
COPY artifacts.json requirements.lock official-downloads.json pins.txt nxdi-source.tar nxdi-source-commit.txt verify-wheelhouse.py /opt/trainium-artifacts/
RUN python /opt/trainium-artifacts/verify-wheelhouse.py /opt/trainium-artifacts \
    && dpkg -i /opt/trainium-artifacts/debs/aws-neuronx-runtime-lib_2.34.10.0-ac18d186d_amd64.deb \
       /opt/trainium-artifacts/debs/aws-neuronx-collectives_2.34.10.0-74eaafac6_amd64.deb \
    && python -m venv /opt/trainium-venv \
    && /opt/trainium-venv/bin/python -m pip install --no-index \
       --find-links=/opt/trainium-artifacts/wheelhouse --require-hashes \
       --requirement /opt/trainium-artifacts/requirements.lock \
    && /opt/trainium-venv/bin/python -m pip check \
    && dpkg-query -W aws-neuronx-runtime-lib aws-neuronx-collectives \
       > /opt/trainium-artifacts/runtime-package-manifest.txt \
    && /opt/trainium-venv/bin/python -m pip freeze \
       > /opt/trainium-artifacts/python-package-manifest.txt
ENV PATH=/opt/trainium-venv/bin:/opt/aws/neuron/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib:/lib/x86_64-linux-gnu:/opt/conda/lib/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
LABEL servingstudio.task="trainium2-torch29" \
      servingstudio.nxdi_commit="4bcdc54bf3b6ccdad490e5cd0680a3dc95270b8e" \
      servingstudio.compiler="2.27.5334.0+f702b353" \
      servingstudio.validation="unvalidated-no-numerical-claim"
ENTRYPOINT ["/opt/trainium-venv/bin/python"]
CMD ["-c", "import importlib.metadata as m; print({n:m.version(n) for n in ('torch','torch-xla','torch-neuronx','neuronx-cc','neuronx-distributed','neuronx-distributed-inference','libneuronxla','nki','transformers','islpy')})"]
