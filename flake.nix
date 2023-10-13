{
  description = "Flake utils demo";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    llm-rs-python = {
      url = "github:mostawesomedude/llm-rs-python";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, llm-rs-python }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        sentence-transformers = pkgs.python310.pkgs.buildPythonPackage rec {
          pname = "sentence-transformers";
          version = "2.2.2";

          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-28YBY7J94hB2yaMNJLW3tvoFFB1ozyVT+pp3v3mikTY=";
          };

          propagatedBuildInputs = with pkgs.python310.pkgs; [
            huggingface-hub nltk scikit-learn scipy sentencepiece tokenizers torch torchvision tqdm transformers
          ];

          doCheck = false;
        };
        py = pkgs.python310.withPackages (ps: [
          llm-rs-python.packages.${system}.default
          ps.faiss sentence-transformers
          ps.fastapi ps.uvicorn
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "housedocs-env";
          buildInputs = [
            py
          ];
        };
      }
    );
}
