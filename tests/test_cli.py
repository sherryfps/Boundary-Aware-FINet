import predict


def test_cli_reports_missing_checkpoint(tmp_path, patterned_image, capsys):
    image = tmp_path / "test.png"
    patterned_image.save(image)
    output = tmp_path / "outputs"
    status = predict.main(["--image", str(image), "--checkpoint", str(tmp_path / "missing.pth"),
                           "--output-dir", str(output)])
    assert status == 1
    assert "checkpoint not found" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_threshold_before_model_load(tmp_path, patterned_image, capsys):
    image = tmp_path / "test.png"
    patterned_image.save(image)
    assert predict.main(["--image", str(image), "--checkpoint", "missing.pth", "--threshold", "nan"]) == 1
    assert "Threshold" in capsys.readouterr().err


def test_cli_never_overwrites_input(tmp_path, patterned_image, capsys):
    image = tmp_path / "probability.png"
    patterned_image.save(image)
    before = image.read_bytes()
    assert predict.main(["--image", str(image), "--checkpoint", "missing.pth", "--output-dir", str(tmp_path)]) == 1
    assert "overwrite the input" in capsys.readouterr().err
    assert image.read_bytes() == before
