using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using Microsoft.ML;

namespace ML.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ExpensePredictionController : ControllerBase
    {
        public class ExpenseData
        {
            [LoadColumn(0)] public float Month;
            [LoadColumn(1)] public float Category;
            [LoadColumn(2)] public float Value;
        }

        public class ExpensePrediction
        {
            [ColumnName("Score")] public float PredictionValue;
        }

        [HttpGet(Name = "next-month-prediction")]
        public ActionResult NextMonthPrediction()
        {
            var mlContext = new MLContext();

            var data = new[]
            {
            new ExpenseData { Month = 1, Category = 1, Value = 500 },
            new ExpenseData { Month = 2, Category = 1, Value = 520 },
            new ExpenseData { Month = 3, Category = 1, Value = 480 },
            new ExpenseData { Month = 4, Category = 1, Value = 510 },
            new ExpenseData { Month = 5, Category = 1, Value = 530 },
        };

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var pipeline = mlContext.Transforms.CopyColumns("Label", "Value")
                    .Append(mlContext.Transforms.Concatenate("Features", "Month", "Category"))
                    .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(dataView);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ExpenseData, ExpensePrediction>(model);

            var novaDespesa = new ExpenseData { Month = 6, Category = 1 };
            var previsao = predictionEngine.Predict(novaDespesa);

            return Ok($"Forecasted expenses for next month: £ {previsao.PredictionValue:F2}");
        }




    }
}
