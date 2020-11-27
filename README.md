<h1>Tradução Neural de Máquina</h1>
<p>Implementação de técnicas para tradução neural de máquina em português com Tensorflow 2.</p>

<h2>Como usar</h2>
<h3><b>pre_processamento.py</b></h3>
<p>
    Recebe : 
    <ul>
        <li>a base de entrada</li>
        <li>a base de saída</li> <li>o número máximo de vocábulos por linha (default: 50)</li>
        <li>o tamanho do vocabulário (default: 50000)</li>
    </ul>
    Retorna : 
    <ul>
        <li>a base de entrada tokenizada</li>
        <li>a base de saída tokenizada</li>
    </ul>
</p>
<br/>
<h3><b>bleu.py</b></h3>
<p>
    Recebe : 
    <ul>
        <li>o tensor com as traduções corretas</li>
        <li>o tensor com as traduções preditas</li>
    </ul>
    Retorna : 
    <ul>
        <li>o bleu médio</li>
    </ul>
</p>