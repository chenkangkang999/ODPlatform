from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("football_portal", "0002_var_review_cases")]
    operations = [migrations.AlterField(model_name="varreviewcase", name="event_type",
                                       field=models.CharField(default="offside", max_length=40))]
