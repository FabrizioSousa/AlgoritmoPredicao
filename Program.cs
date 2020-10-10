using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Extensions.Configuration;
using System.Data.SqlClient;

using System.Threading.Tasks;
using PredicaoBiblioteca;
using Microsoft.Azure.Storage;
using Microsoft.Azure.Storage.Blob;

namespace TesteRegLogMLNet
{
    class Program
    {
        static readonly string _dataPath =
            "model.zip";

        private static string _SqlConnection;

        static async Task Main(string[] args)
        {
            //Console.WriteLine("\nPredição de inadimplência\n");
            //MLContext mlContext = new MLContext();

            //TrainTestData splitDataView = LoadData(mlContext);

            //ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //// 1. load data and create data pipeline

            //Evaluate(mlContext, model, splitDataView.TestSet);

            //UseModelWithSingleItem(mlContext, model);

            //Console.WriteLine();
            //Console.WriteLine("=============== End of process ===============");
            //Console.ReadLine();

            var builder = new ConfigurationBuilder().SetBasePath(Directory.GetCurrentDirectory()).AddJsonFile("config.json");
            var configuration = builder.Build();
            _SqlConnection = configuration["connectionString"];
            var items = File.ReadAllLines("./Pesquisa sobre Gastos Pessoais.csv").Skip(1).Select(line => line.Split(';'))
                .Select(i =>
                new Usuario
                {
                    ID = Convert.ToInt32(i[0]),
                    Idade = Convert.ToInt32(i[1]),
                    Sexo = Convert.ToInt32(i[2]),
                    Escolaridade = Convert.ToInt32(i[3]),
                    Fl_InadimplentePassado = Convert.ToInt32(i[4]),
                    Renda = Convert.ToInt32(i[5]),
                    Desp_Fixas = Convert.ToInt32(i[6]),
                    Desp_Variaveis = Convert.ToInt32(i[7]),
                    Contas_Atrasadas = Convert.ToInt32(i[8]),


                });
            using (var connection = new SqlConnection(_SqlConnection))
            {
                connection.Open();
                var insertCommand = @"Insert into PredicaoInadimplencia.dbo.Usuario values
                (@ID, @Idade,@Sexo,@Escolaridade,@Fl_InadimplentePassado,@Renda,@Desp_Fixas,@Desp_Variaveis,@Contas_Atrasadas)";


                foreach (var item in items)
                {
                    var command = new SqlCommand(insertCommand, connection);
                    command.Parameters.AddWithValue("@ID", item.ID);
                    command.Parameters.AddWithValue("@Idade", item.Idade);
                    command.Parameters.AddWithValue("@Sexo", item.Sexo);
                    command.Parameters.AddWithValue("@Escolaridade", item.Escolaridade);
                    command.Parameters.AddWithValue("@Fl_InadimplentePassado", item.Fl_InadimplentePassado);
                    command.Parameters.AddWithValue("@Renda", item.Renda);
                    command.Parameters.AddWithValue("@Desp_Fixas", item.Desp_Fixas);
                    command.Parameters.AddWithValue("@Desp_Variaveis", item.Desp_Variaveis);
                    command.Parameters.AddWithValue("@Contas_Atrasadas", item.Contas_Atrasadas);

                    command.ExecuteNonQuery();
                }
            }
            var data = new List<Usuario>();
            using (var connection = new SqlConnection(_SqlConnection))
            {
                connection.Open();
                var selectCommand = "Select * from dbo.Usuario";
                var sqlCommand = new SqlCommand(selectCommand, connection);
                var reader = sqlCommand.ExecuteReader();

                while (reader.Read())
                {
                    data.Add(new Usuario
                    {
                        ID = Convert.ToInt32(reader.GetValue(0)),
                        Idade = Convert.ToInt32(reader.GetValue(1)),
                        Sexo = Convert.ToInt32(reader.GetValue(2)),
                        Escolaridade = Convert.ToInt32(reader.GetValue(3)),
                        Fl_InadimplentePassado = Convert.ToInt32(reader.GetValue(4)),
                        Renda = Convert.ToInt32(reader.GetValue(5)),
                        Desp_Fixas = Convert.ToInt32(reader.GetValue(6)),
                        Desp_Variaveis = Convert.ToInt32(reader.GetValue(7)),
                        Contas_Atrasadas = Convert.ToInt32(reader.GetValue(8)),
                    });
                }
                Console.WriteLine("\nPredição de inadimplência\n");
                var mlContext = new MLContext();

                IDataView dataView = mlContext.Data.LoadFromEnumerable(data);

                var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

                //var TrainSet = mlContext.Data.CreateEnumerable<Usuario>(split.TrainSet, reuseRowObject: false);

                //var testSet = mlContext.Data.CreateEnumerable<Usuario>(split.TestSet, reuseRowObject: false);

                //var Preview = split.TrainSet.Preview();
                var Idade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
          { new InputOutputColumnPair("Idade", "Idade") });
                var Sexo = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Sexo", "Sexo") });
                var Escolaridade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Escolaridade", "Escolaridade") });
                var Fl_InadimplentePassado = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Fl_InadimplentePassado", "Fl_InadimplentePassado") });
                var Renda = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Renda", "Renda") });
                var Desp_Fixas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Desp_Fixas", "Desp_Fixas") });
                var Desp_Variaveis = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Desp_Variaveis", "Desp_Variaveis") });
                var Contas_Atrasadas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            { new InputOutputColumnPair("Contas_Atrasadas", "Contas_Atrasadas") });


                var c = mlContext.Transforms.Concatenate("Features", new[]
                  { "Idade", "Sexo", "Escolaridade"
                    , "Fl_InadimplentePassado", "Renda", "Desp_Fixas", "Desp_Variaveis", "Contas_Atrasadas"
           });
                var dataPipe = Idade.Append(Idade).Append(Sexo)
                    .Append(Escolaridade).Append(Fl_InadimplentePassado).Append(Renda)
                    .Append(Desp_Fixas).Append(Desp_Variaveis)
                    .Append(Contas_Atrasadas).Append(c);

                var estimator = dataPipe.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

                Console.WriteLine("=============== Create and Train the Model ===============");
                var model = estimator.Fit(split.TrainSet);

                var storageAccount = CloudStorageAccount.Parse(configuration["blobConnectionString"]);

                var client = storageAccount.CreateCloudBlobClient();
                var container = client.GetContainerReference("models");

                var blob = container.GetBlockBlobReference(_dataPath);

                using (var stream = File.Create(_dataPath))
                {
                    mlContext.Model.Save(model, dataView.Schema, stream);
                }

                await blob.UploadFromFileAsync(_dataPath);
            }



        }



    }




    //public static TrainTestData LoadData(MLContext mlContext)
    //{
    //    IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(_dataPath, hasHeader: true, separatorChar: ';');

    //    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.45);

    //    return splitDataView;


    //}

    //    public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    //    {
    //        var Idade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //      { new InputOutputColumnPair("Idade", "Idade") });
    //        var Sexo = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Sexo", "Sexo") });
    //        var Escolaridade = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Escolaridade", "Escolaridade") });
    //        var Fl_InadimplentePassado = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Fl_InadimplentePassado", "Fl_InadimplentePassado") });
    //        var Renda = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Renda", "Renda") });
    //        var Desp_Fixas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Desp_Fixas", "Desp_Fixas") });
    //        var Desp_Variaveis = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Desp_Variaveis", "Desp_Variaveis") });
    //        var Contas_Atrasadas = mlContext.Transforms.Categorical.OneHotEncoding(new[]
    //    { new InputOutputColumnPair("Contas_Atrasadas", "Contas_Atrasadas") });


    //        var c = mlContext.Transforms.Concatenate("Features", new[]
    //          { "Idade", "Sexo", "Escolaridade"
    //                , "Fl_InadimplentePassado", "Renda", "Desp_Fixas", "Desp_Variaveis", "Contas_Atrasadas"
    //       });
    //        var dataPipe = Idade.Append(Idade).Append(Sexo)
    //            .Append(Escolaridade).Append(Fl_InadimplentePassado).Append(Renda)
    //            .Append(Desp_Fixas).Append(Desp_Variaveis)
    //            .Append(Contas_Atrasadas).Append(c);

    //        var estimator = dataPipe.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    //        Console.WriteLine("=============== Create and Train the Model ===============");
    //        var model = estimator.Fit(splitTrainSet);
    //        Console.WriteLine("=============== End of training ===============");
    //        Console.WriteLine();

    //        return model;
    //    }

    //    public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    //    {
    //        Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    //        IDataView predictions = model.Transform(splitTestSet);

    //        CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    //        Console.WriteLine();
    //        Console.WriteLine("Model quality metrics evaluation");
    //        Console.WriteLine("--------------------------------");
    //        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    //        Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    //        Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    //        Console.WriteLine("=============== End of model evaluation ===============");
    //    }

    //    private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
    //    {
    //        PredictionEngine<ModelInput, ModelOutput> predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

    //        ModelInput sampleStatement = new ModelInput
    //        {
    //            Idade = 20,
    //            Sexo = 1,
    //            Escolaridade = 2,
    //            Fl_InadimplentePassado = 1,
    //            Renda = 5000,
    //            Desp_Fixas = 1000,
    //            Desp_Variaveis = 3000,
    //            Contas_Atrasadas = 1
    //        }; 
    //        var resultPrediction = predictionFunction.Predict(sampleStatement);

    //        Console.WriteLine();
    //        Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    //        Console.WriteLine();
    //        Console.WriteLine($"Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    //        Console.WriteLine("=============== End of Predictions ===============");
    //        Console.WriteLine();
    //    }

    //    // Main
    //} // Program

    class ModelInput
    {
        [LoadColumn(0)]
        public int ID;

        [LoadColumn(1)]
        public int Idade;
        [LoadColumn(2)]
        public int Sexo;
        [LoadColumn(3)]
        public int Escolaridade;
        [LoadColumn(4), ColumnName("Label")]
        public bool Fl_Inadimplente;
        [LoadColumn(5)]
        public int Fl_InadimplentePassado;

        [LoadColumn(6)]
        public int Renda;
        [LoadColumn(7)]
        public int Desp_Fixas;
        [LoadColumn(8)]
        public int Desp_Variaveis;
        [LoadColumn(9)]
        public int Contas_Atrasadas;


    }

    class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }


        public float Score { get; set; }

        public float Probability { get; set; }
    }
}


