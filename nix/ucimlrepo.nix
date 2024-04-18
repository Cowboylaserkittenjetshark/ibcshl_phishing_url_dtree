{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  pandas,
  certifi,
}:
buildPythonPackage rec {
  pname = "ucimlrepo";
  version = "0.0.6";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "uci-ml-repo";
    repo = "ucimlrepo";
    rev = "c962c8bf547c4092c334c56fd47614b4099078b3";
    hash = "sha256-k9FEKfzyoW+sSfw5UD+sBCEUycYNqPYjDeWlyaWyxzk=";
  };

  nativeBuildInputs = [
    setuptools
  ];

  propagatedBuildInputs = [
    pandas
    certifi
  ];

  pythonImportsCheck = ["ucimlrepo"];

  meta = {
    description = "Package to easily import datasets from the UC Irvine Machine Learning Repository into scripts and notebooks.";
    homepage = "https://github.com/uci-ml-repo/ucimlrepo/blob/main/pyproject.toml";
    maintainers = with lib.maintainers; [cowboylaserkittenjetshark];
    license = lib.licenses.mit;
  };
}
