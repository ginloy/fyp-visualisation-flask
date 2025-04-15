{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    LD_LIBRARY_PATH = builtins.getEnv "NIX_LD_LIBRARY_PATH";
  }
