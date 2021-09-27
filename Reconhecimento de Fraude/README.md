# EN/en

The Credit Card Dataset can be accessed via the hyperlink below:

[Credit Card]("https://www.kaggle.com/mlg-ulb/creditcardfraud")

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 

Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

##Libraries

The following libraries/packages need to be installed before running the project:

```bash
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
```


# PT/br

O Credit Card Dataset pode ser acessado através do hyperlink abaixo:

[Credit Card]("https://www.kaggle.com/mlg-ulb/creditcardfraud")

É importante que as empresas de cartão de crédito sejam capazes de reconhecer transações fraudulentas com cartão de crédito para que os clientes não sejam cobrados por itens que não compraram.

Ele contém apenas variáveis de entrada numéricas que são o resultado de uma transformação PCA. Infelizmente, devido a questões de confidencialidade, não podemos fornecer os recursos originais e mais informações básicas sobre os dados. Os recursos V1, V2,… V28 são os componentes principais obtidos com o PCA, os únicos recursos que não foram transformados com o PCA são 'Tempo' e 'Quantidade'. O recurso 'Tempo' contém os segundos decorridos entre cada transação e a primeira transação no conjunto de dados. 

O recurso 'Amount' é o Amount da transação, este recurso pode ser usado como exemplo de aprendizagem dependente de custos. O recurso 'Classe' é a variável de resposta e assume o valor 1 em caso de fraude e 0 em caso contrário.

## Bibliotecas

As seguintes bibliotecas /pacotes precisam ser instalados antes de rodar o projeto:

```bash
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
```

