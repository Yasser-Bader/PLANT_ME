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

                        document.querySelector('.printt').innerHTML =`نبات البطاطس مصاب باللفحة المبكره
                        الأعراض
                        1-تظهر أولى أعراض المرض على هيئة بقع داكنة مع نمو متحد المركز وهالات صفراء على الأوراق.
                        2-متبوع بجفاف الأوراق وتساقطها .
                        3-الثمار قد تبدأ بالتعفن ثم تسقط في نهاية الأمر.
                        تحدث أعراض اللفحة المبكرة على أوراق الشجر القديمة والسيقان ، والثمار . تظهر بقع ذات لون رمادي مائل إلى البني على أوراق الشجر ، وتنمو تدريجياً بطريقة مركزية حول مركز واضح- مما يشكل المظهر المميز الذي يشبه " هدف الرماية " . تكون هذه البقع محاطة بهالة صفراء زاهية ومع تقدم المرض ، قد تتحول الأوراق بأكملها إلى اللون الأصفر الناجم عن نقص الكلوروفيل وتتساقط ، مما يؤدي إلى تساقط الأوراق بشكل ملحوظ . وعندما تموت الأوراق وتسقط ، تصبح الثمار أكثر عرضة لحروق الشمس . ويظهر نفس النوع من البقع ذات المركز الواضح على السيقان والفواكه. ويبدأ تعفن الفواكه وقد تتساقط في النهاية
                        اجراءت وقائية
                        يجب عليك استخدام بذور معتمدة أو خالية من مسببات الأمراض. 
                        والبحث عن الأصناف المقاومة للمرض.
                        يجب زراعة البذور أو الشتلات على مهاد مرتفع ومحروث لتحسين الصرف.
                        وتأكد من توجيه الصفوف في اتجاه الرياح الرئيسية وتجنب المناطق المظللة.
                        وحافظ على مساحة مناسبة بين النباتات للسماح للمظلة الخضرية أن تجف بسرعة بعد هطول الأمطار أو الري.
                        كما يجب وضع المهاد على التربة لمنع النباتات من لمس التربة
                        وكذلك مراقبة الحقول بحثا عن علامات المرض ، خاصة أثناء الطقس الرطب.
                        وإزالة الأوراق السفلية القريبة جداً من التربة.
                        وإزالة الأوراق التي تمت ملاحظة الأعراض عليها وتدميرها.
                        وكذلك الحفاظ على النباتات قوية ومخضرة بإضافة ما يكفي من المغذيات.
                        وكذلك استخدام العصي الخشبية للحفاظ على النباتات منتصبة ومستقيمة.
                        استخدام نظام الري بالتنقيط للحد من بلل الأوراق.
                        وقم بالري في الصباح حتى تحافظ على النباتات جافة خلال النهار.
                        وقم أيضاً بمكافحة الحشائش القابلة للإصابة في الحقل ومن حوله.
                        أما بعد الحصاد ، فقم بإزالة المخلفات النباتية وحرقها ( ولا تقم بتحويلها إلى سماد بطريقة الكومبوست ).
                        بدلاً من ذلك ، يمكنك استخدام الحرث العميق لدفن المخلفات عميقاً في التربة ( أكثر من 45 سم ).
                        وتخطيط وتنفيذ تناوب المحاصيل لمدة 2 أو 3 سنوات مع المحاصيل غير المضيفة أو المعرضة للإصابة.
                        وقم بتخزين الثمار في درجات حرارة باردة وفي مواقع جيدة التهوية.
                        المكافحة الكيميائية
                        1-مبيدات الفطرية proineb 70.0 WP
                        2-مبيدات البكتريا copper oxychloride 50.0 WP
                        3-مبيدات الفطريات mancozeb 75.0 WP
                        سبب المرض
                        تنجم الأعراض عن فطر ألترناريا سولاني ، وهي فطريات تعيش على بقايا المحاصيل المصابة في التربة أو على العوائل البديلة . كما قد تكون البذور المشتراة أو الشتلات ملوثة بالفعل . وغالبا ما تصاب الأوراق السفلي عند ملامستها للتربة الملوثة . وتعد درجات الحرارة الدافئة ( 29-24 درجة مئوية ) والرطوبة العالية ( 90 % ) ظروفاً مواتية لتطور المرض . ولذلك تعمل الفترات الرطبة الطويلة ( أو الطقس الرطب / الجاف بالتناوب ) على تحسين إنتاج الأبواغ الفطرية ، والتي قد تنتشر عن طريق الرياح أو تناثر قطرات المطر أو الري العلوي . تكون الثمار التي يتم حصادها ولونها أخضر أو في الظروف الرطبة معرضة بشكل خاص للعدوى . وغالباً ما ينتشر المرض بشدة بعد فترة من الأمطار الغزيرة ، ويعد من الأمراض المدمرة خصوصاً في المناطق الاستوائية وشبه الاستوائية .`

                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس مصاب باللفحة المبكرة</h2>
                        <h2>🍂 الأعراض</h2>
                        1-تظهر أولى أعراض المرض على هيئة بقع داكنة مع نمو متحد المركز وهالات صفراء على الأوراق.<br>
                        2-متبوع بجفاف الأوراق وتساقطها .<br>
                        3-الثمار قد تبدأ بالتعفن ثم تسقط في نهاية الأمر.<br>
                        تحدث أعراض اللفحة المبكرة على أوراق الشجر القديمة والسيقان ، والثمار . تظهر بقع ذات لون رمادي مائل إلى البني على أوراق الشجر ، وتنمو تدريجياً بطريقة مركزية حول مركز واضح- مما يشكل المظهر المميز الذي يشبه " هدف الرماية " . تكون هذه البقع محاطة بهالة صفراء زاهية ومع تقدم المرض ، قد تتحول الأوراق بأكملها إلى اللون الأصفر الناجم عن نقص الكلوروفيل وتتساقط ، مما يؤدي إلى تساقط الأوراق بشكل ملحوظ . وعندما تموت الأوراق وتسقط ، تصبح الثمار أكثر عرضة لحروق الشمس . ويظهر نفس النوع من البقع ذات المركز الواضح على السيقان والفواكه. ويبدأ تعفن الفواكه وقد تتساقط في النهاية
                        <br>
                        <h2>اجراءات وقائية</h2>
                        يجب عليك استخدام بذور معتمدة أو خالية من مسببات الأمراض. <br>
                        والبحث عن الأصناف المقاومة للمرض.<br>
                        يجب زراعة البذور أو الشتلات على مهاد مرتفع ومحروث لتحسين الصرف.<br>
                        وتأكد من توجيه الصفوف في اتجاه الرياح الرئيسية وتجنب المناطق المظللة.<br>
                        وحافظ على مساحة مناسبة بين النباتات للسماح للمظلة الخضرية أن تجف بسرعة بعد هطول الأمطار أو الري.<br>
                        كما يجب وضع المهاد على التربة لمنع النباتات من لمس التربة.<br>
                        وكذلك مراقبة الحقول بحثا عن علامات المرض ، خاصة أثناء الطقس الرطب.<br>
                        وإزالة الأوراق السفلية القريبة جداً من التربة.<br>
                        وإزالة الأوراق التي تمت ملاحظة الأعراض عليها وتدميرها.<br>
                        وكذلك الحفاظ على النباتات قوية ومخضرة بإضافة ما يكفي من المغذيات.<br>
                        وكذلك استخدام العصي الخشبية للحفاظ على النباتات منتصبة ومستقيمة.<br>
                        استخدام نظام الري بالتنقيط للحد من بلل الأوراق.<br>
                        وقم بالري في الصباح حتى تحافظ على النباتات جافة خلال النهار.<br>
                        وقم أيضاً بمكافحة الحشائش القابلة للإصابة في الحقل ومن حوله.<br>
                        أما بعد الحصاد ، فقم بإزالة المخلفات النباتية وحرقها ( ولا تقم بتحويلها إلى سماد بطريقة الكومبوست ).<br>
                        بدلاً من ذلك ، يمكنك استخدام الحرث العميق لدفن المخلفات عميقاً في التربة ( أكثر من 45 سم ).<br>
                        وتخطيط وتنفيذ تناوب المحاصيل لمدة 2 أو 3 سنوات مع المحاصيل غير المضيفة أو المعرضة للإصابة.<br>
                        وقم بتخزين الثمار في درجات حرارة باردة وفي مواقع جيدة التهوية.<br>

                        <h2>⚗ المكافحة الكيميائية</h2>
                        1-مبيدات الفطرية proineb 70.0 WP<br>
                        2-مبيدات البكتريا copper oxychloride 50.0 WP<br>
                        3-مبيدات الفطريات mancozeb 75.0 WP<br>

                        <h2> سبب المرض</h2>
                        تنجم الأعراض عن فطر ألترناريا سولاني ، وهي فطريات تعيش على بقايا المحاصيل المصابة في التربة أو على العوائل البديلة . كما قد تكون البذور المشتراة أو الشتلات ملوثة بالفعل . وغالبا ما تصاب الأوراق السفلي عند ملامستها للتربة الملوثة . وتعد درجات الحرارة الدافئة ( 29-24 درجة مئوية ) والرطوبة العالية ( 90 % ) ظروفاً مواتية لتطور المرض . ولذلك تعمل الفترات الرطبة الطويلة ( أو الطقس الرطب / الجاف بالتناوب ) على تحسين إنتاج الأبواغ الفطرية ، والتي قد تنتشر عن طريق الرياح أو تناثر قطرات المطر أو الري العلوي . تكون الثمار التي يتم حصادها ولونها أخضر أو في الظروف الرطبة معرضة بشكل خاص للعدوى . وغالباً ما ينتشر المرض بشدة بعد فترة من الأمطار الغزيرة ، ويعد من الأمراض المدمرة خصوصاً في المناطق الاستوائية وشبه الاستوائية .
                        `

                    }if (data[class_idx] == "Pepper,_bell___Bacterial_spot") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الفلفل مصاب ببقعة البكتريا</h2>`
                        document.querySelector('.printt').innerHTML =`نبات الفلفل مصاب ببقعة البكتريا`
                    } if (data[class_idx] == "Pepper,_bell___healthy") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الفلفل غير مصاب (صحي)</h2>`
                        document.querySelector('.printt').innerHTML =`نبات الفلفل غير مصاب (صحي)`
                    } if (data[class_idx] == "Potato___healthy") {
                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس غير مصاب (صحي)</h2>`
                        document.querySelector('.printt').innerHTML =`نبات البطاطس غير مصاب (صحي)`
                    } if (data[class_idx] == "Potato___Late_blight") {
                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس مصاب باللفحة المتأخرة </h2>`
                        document.querySelector('.printt').innerHTML =`نبات البطاطس مصاب باللفحة المتأخرة`
                    } if (data[class_idx] == "Tomato___Target_Spot") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطماطم مصاب بالتبقع </h2>`
                        document.querySelector('.printt').innerHTML =`نبات الطماطم مصاب بالتبقع`
                    } if (data[class_idx] == "Tomato___Tomato_mosaic_virus") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطاطم مصاب بفيروس فسيفساء</h2>`
                        document.querySelector('.printt').innerHTML =`نبات الطاطم مصاب بفيروس فسيفساء`
                    } if (data[class_idx] == "Tomato___Tomato_Yellow_Leaf_Curl_Virus") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطماطم مصاب بتجعد الأوراق الصفراء</h2>`
                        document.querySelector('.printt').innerHTML =`نبات الطماطم مصاب بتجعد الأوراق الصفراء`
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