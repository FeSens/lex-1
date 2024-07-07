from encodec.model import EncodecModel
import torchaudio
import torch

tokens = torch.load('./encoded/lex_ai_sam_altman_2.pt').to("mps")
print(tokens.shape)
# def display_audio(input, label="Decoded Audio:"):
with torch.no_grad():
    input = tokens[:10]#[:10, :, :].transpose(0,1).flatten(start_dim=1, end_dim=-1).unsqueeze(0)
    print(input.shape)
    model = EncodecModel.encodec_model_24khz().to("mps")
    decoded_audio = model.decode([[input, None]]).transpose(0,1).flatten(start_dim=1, end_dim=-1)
    print(decoded_audio.shape)
    torchaudio.save("decoded_audio.mp3", decoded_audio.cpu(), 24000)
