# Transformer Architecture

```mermaid
graph LR
    %% Input Embeddings
    IE[Input Embedding] --> PE[Positional Encoding]
    
    %% Encoder Stack (repeated 6-12 layers)
    PE --> EN1[Add & Norm 1]
    EN1 --> MHA1[Multi-Head<br/>Attention]
    MHA1 --> EN2[Add & Norm 2]
    EN2 --> FFN1[Feed Forward]
    FFN1 --> EN3[Add & Norm 3]
    EN3 --> EN_OUT[Encoder Output]
    
    %% Decoder Stack (repeated 6-12 layers)  
    OE[Output Embedding] --> PE2[Positional<br/>Encoding]
    PE2 --> DE1[Add & Norm 1]
    DE1 --> MHA_DEC[Masked Multi-Head<br/>Attention]
    MHA_DEC --> DE2[Add & Norm 2]
    DE2 --> MHA_ENC[Multi-Head Attention<br/>(Encoder-Decoder)]
    MHA_ENC --> DE3[Add & Norm 3]
    DE3 --> FFN2[Feed Forward]
    FFN2 --> DE4[Add & Norm 4]
    DE4 --> DEC_OUT[Decoder Output]
    
    %% Final layers
    EN_OUT --> MHA_ENC
    DEC_OUT --> LIN[Linear Layer]
    LIN --> SOFT[Softmax]
    
    %% Styling matching your image
    classDef encoder fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef decoder fill:#f3e5f5,stroke:#4a148c,stroke-width:3px  
    classDef embedding fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px
    classDef output fill:#fff3e0,stroke:#e65100,stroke-width:3px
    
    class IE,PE,EN1,MHA1,EN2,FFN1,EN3,EN_OUT encoder
    class OE,PE2,DE1,MHA_DEC,DE2,MHA_ENC,DE3,FFN2,DE4,DEC_OUT decoder
    class IE,OE,PE,PE2 embedding
    class LIN,SOFT output

```