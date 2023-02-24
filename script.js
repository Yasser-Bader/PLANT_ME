let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage')
let img = document.getElementById('image')
let boxResult = document.querySelector('.box-result')
let confidence = document.querySelector('.confidence')
let pconf = document.querySelector('.box-result p')
/*
$("textarea").each(function () {
    this.setAttribute("style", "height:" + (this.scrollHeight) + "px;overflow-y:hidden;");
  }).on("input", function () {
    this.style.height = 0;
    this.style.height = (this.scrollHeight) + "px";
  });*/
        
        let progressBar = 
            new ProgressBar.Circle('#progress', {
            color: 'limegreen',
            strokeWidth: 10,
            duration: 2000, // milliseconds
            easing: 'easeInOut'
        });

        async function fetchData(){
            let response = await fetch('./class_indices.json');
            let data = await response.json();
            data = JSON.stringify(data);
            data = JSON.parse(data);
            return data;
        }

         // here the data will be return.
        

        // Initialize/Load model
        async function initialize() {
            let status = document.querySelector('.init_status')
            status.innerHTML = 'جار التحميل .... <span class="fa fa-spinner fa-spin"></span>'
            model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
            status.innerHTML = 'تم تحميل النموذج بنجاح  <span class="fa fa-check"></span>'
        }

        async function predict() {
            // Function for invoking prediction
            let img = document.getElementById('image')
            let offset = tf.scalar(255)
            let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
            let tensorImg_scaled = tensorImg.div(offset)
            prediction = await model.predict(tensorImg_scaled).data();
           
            fetchData().then((data)=> 
                {
                    predicted_class = tf.argMax(prediction)
                    
                    class_idx = Array.from(predicted_class.dataSync())[0]
                    document.querySelector('.pred_class').innerHTML = data[class_idx]
                    document.querySelector('.inner').innerHTML = `متأكد ${parseFloat(prediction[class_idx]*100).toFixed(2)}%`
                    console.log(data)
                    console.log(data[class_idx])
                    console.log(prediction)

                    progressBar.animate(prediction[class_idx]-0.005); // percent

                    pconf.style.display = 'block'

                    confidence.innerHTML = Math.round(prediction[class_idx]*100)
                    if(data[class_idx] == "Potato___Early_blight")
                    {

                        document.querySelector('.printt').innerHTML =`نباتك مصاب باللفحة المبكرة في البطاطا
الأعراض:
تحدث الإصابة باللفحة المبكرة في البطاطا بسبب الفطر. حيث يؤثر المرض على الأوراق والسيقان والدرنات ويمكن أن يقلل المحصول وحجم الدرنات وقابلية تخزين الدرنات وجودة السوق الطازجة ودرنات المعالجة وإمكانية تسويق المحصول.

شدة اللفحة المبكرة على البطاطا
في معظم مناطق الإنتاج ، تحدث اللفحة المبكرة سنويًا إلى حد ما.
و تعتمد شدة اللفحة المبكرة على :

معدل رطوبة الأوراق من المطر أو الندى أو الري .
الحالة التغذوية لأوراق الشجر.
وقابلية الصنف للإصابة .

الوقاية ومكافحة اللفحة المبكرة على البطاطا

زراعة أصناف متأخرة النضج ذات قابلية أقل للإصابة باللفحة المبكرة. المقاومة مرتبطة بنضج النبات والأصناف المبكرة النضج أكثر عرضة.
توقيت الري لتقليل مدة رطوبة الأوراق أثناء الطقس الغائم وإتاحة الوقت الكافي لتجف الأوراق قبل حلول الظلام.
تجنب نقص الفسفور والنيتروجين.
قم بفحص الحقول بانتظام للتأكد من العدوى التي تبدأ بعد أن يصل ارتفاع النباتات إلى 12 بوصة. و انتبه بشكل خاص إلى حواف الحقول المجاورة للحقول المزروعة بالبطاطا في العام السابق.
تناوب مبيدات الفطريات الورقية المتخصصة .
التخلص من عروش النباتات قبل الحصاد بأسبوعين إلى ثلاثة أسابيع .
تجنب إحداث الجروح أثناء الحصاد.
قم بتخزين الدرنات في ظروف تعزز التئام الجروح (الهواء النقي ، 95 إلى 99 في المائة من الرطوبة النسبية ، ودرجات حرارة من 55 إلى 60 فهرنهايت) لمدة أسبوعين إلى ثلاثة أسابيع بعد الحصاد.
بعد التئام الجروح ، قم بتخزين الدرنات في مكان مظلم وجاف وجيد التهوية مبرد تدريجيًا إلى درجة حرارة مناسبة للسوق المطلوب.
الدورة الزراعية :تناوب الحقول على المحاصيل غير المضيفة لمدة ثلاث سنوات على الأقل (تناوب المحاصيل من ثلاث إلى أربع سنوات).
القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.`

                        document.querySelector('.print').innerHTML =`<h2>🍂 الأعراض</h2>
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.

                        <h2>اجراءات وقائية</h2>
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.

                        <h2>⚗ المكافحة الكيميائية</h2>
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.

                        <h2> سبب المرض</h2>
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.
                        القضاء على عوائل الحشائش مثل الباذنجان المشعر لتقليل اللقاح للزراعة المستقبلية.
                        `

                    }else{
                      document.querySelector('.print').innerHTML = "no no"
                    }
                }
            );
            
        }

        

        fileUpload.addEventListener('change', function(e){
            
            let uploadedImage = e.target.value
            if (uploadedImage){
                document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","")
                document.getElementById("choose-text-1").innerText = "تغيير الصورة المختارة"
                document.querySelector(".success-1").style.display = "inline-block"

                let extension = uploadedImage.split(".")[1]
                if (!(["doc","docx","pdf"].includes(extension))){
                    document.querySelector(".success-1 i").style.border = "1px solid limegreen"
                    document.querySelector(".success-1 i").style.color = "limegreen"
                }else{
                    document.querySelector(".success-1 i").style.border = "1px solid rgb(25,110,180)"
                    document.querySelector(".success-1 i").style.color = "rgb(25,110,180)"
                }
            }
            let file = this.files[0]
            if (file){
                boxResult.style.display = 'block'
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.addEventListener("load", function(){
                    
                    img.style.display = "block"
                    img.setAttribute('src', this.result);
                });
            }

            else{
            img.setAttribute("src", "");
            }

            initialize().then( () => { 
                predict()
            })
        })