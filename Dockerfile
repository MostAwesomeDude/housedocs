FROM nixos/nix

WORKDIR /app/
COPY flake.nix flake.lock /app/

RUN nix build --extra-experimental-features flakes --extra-experimental-features nix-command

COPY main.py /app/

EXPOSE 8000
CMD ["/app/result/bin/uvicorn", "--host", "0.0.0.0", "main:app"]
