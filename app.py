from flask import Flask, render_template, request, flash, url_for
import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import sklearn
uploadedimges="static/userUpload"
result="result.html" 
healthy_heart="static/assest/Healthy_heart.jpg"
unhealtht_heart="static/assest/Unhealthy_heart.jpg"
#--------------------------------------------------------------------

#--------------------------------loading models-----------------------------
heart_model=pickle.load(open("models\Heart_Disease.pkl", "rb"))
diabetes_model=pickle.load(open("models\Diabetes_disease.pkl", "rb"))
lungs_model=load_model("models\Lungs_disease.h5")
skin_model=load_model("models\Skin_disease4.h5")
brain_model=load_model("models/brain_disease.h5")
#print("models loaded")
#----------------------------------------------------------------------------
def brain_models_prediction(brain_Images):
    test_image=load_img(brain_Images, target_size=(224, 224))
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
    result = brain_model.predict(test_image).round(3) # predict diseased palnt or not
    #print('@@ Raw result = ', result)
    pred=np.argmax(result)
    if pred == 0:
        #print("GLIOMA")
        data={
        "disease":"Glioma Brain Disease",
        "abstraction":"Glioma is a type of tumor that originates from glial cells, which are supportive cells in the central nervous system (CNS). Gliomas are the most common type of primary brain tumor, accounting for about 80% of all cases. They can occur in any part of the CNS, including the brain, spinal cord, and nerve roots.<br>Diagnosis of glioma is typically made using a combination of imaging tests, such as MRI or CT scans, and biopsies, which involve removing a small sample of tissue for examination under a microscope.<br>The prognosis for glioma depends on the grade of the tumor, the patient's age and overall health, and the response to treatment. Low-grade gliomas are generally slow-growing and can be managed with surgery, radiation therapy, and chemotherapy. High-grade gliomas are more aggressive and have a poorer prognosis. However, advances in treatment have improved survival rates for patients with high-grade gliomas in recent years.<br>If you are concerned that you or someone you know may have glioma, it is important to see a doctor for diagnosis and treatment. Early diagnosis and treatment can improve the chances of a good outcome.",
        "symptoms_define":"The symptoms of glioma can vary depending on the location and size of the tumor. Some common symptoms include:",
        "symptoms_list":"<li>Headaches: These may be constant, intermittent, or worse in the morning.</li><li>Seizures: These are more common with tumors located in the temporal lobe of the brain</li><li>Focal neurological deficits: These can include weakness or paralysis on one side of the body, difficulty with speech or vision, or problems with balance or coordination.</li><li>Cognitive changes: These can include memory problems, difficulty with concentration, or changes in personality or behavior.</li>",
        "treatment":"Treatment options for glioma depend on the grade of the tumor, the patient's age and overall health, and the location of the tumor. Treatment options may include: <li>Surgery: This is the primary treatment for low-grade gliomas. The goal of surgery is to remove as much of the tumor as possible without causing damage to healthy brain tissue.</li><li>Radiation therapy: This uses high-energy beams to kill cancer cells. Radiation therapy may be used after surgery to help reduce the risk of the tumor coming back, or it may be used as the primary treatment for high-grade gliomas.</li><li>Chemotherapy: This uses drugs to kill cancer cells throughout the body. Chemotherapy may be used in conjunction with radiation therapy or on its own for high-grade gliomas.</li><li>Targeted therapy: This uses drugs that specifically target cancer cells. Targeted therapy is a newer treatment option for glioma and is still under investigation.</li>"
    }
        return data # if index 0 burned leaf
    elif pred==1:
        #print("MENIGIOMA")
        data={
        "disease":"Meningioma Brain Disease",
        "abstraction":"A meningioma is a type of tumor that arises from the meninges, which are the membranes that surround the brain and spinal cord. These tumors are the most common type of primary brain tumor, accounting for approximately 30% of all cases. They are typically benign (noncancerous) and slow-growing, and they rarely spread to other parts of the body.",
        "symptoms_define":"The symptoms of meningioma depend on the location and size of the tumor. Some common symptoms include:",
        "symptoms_list":"<li>Headaches</li><li>Seizures</li><li>Vision problems</li><li>Numbness or weakness in the arms or legs</li><li>Difficulty with balance or coordination</li><li>Memory problems</li><li>Changes in personality or behavior</li>",
        "treatment":"Treatment options for meningioma depend on the location and size of the tumor, the patient's age and overall health, and the presence of any symptoms. Treatment options may include: <li>Observation: For small, slow-growing tumors that are not causing any symptoms, observation may be the best course of treatment. The tumor will be monitored with regular imaging tests to check for any changes.</li><li>Surgery: This is the primary treatment for meningiomas that are causing symptoms or that are growing rapidly. The goal of surgery is to remove as much of the tumor as possible without causing damage to healthy brain tissue.</li><li>Radiation therapy: This uses high-energy beams to kill cancer cells. Radiation therapy may be used after surgery to help reduce the risk of the tumor coming back, or it may be used as the primary treatment for meningiomas that are difficult to remove surgically.</li><li>Stereotactic radiosurgery: This is a type of radiation therapy that uses very precisely focused beams to target the tumor. Stereotactic radiosurgery is often used for meningiomas that are located in difficult-to-reach areas of the brain.</li>"
    }
        return data
    elif pred==2:
        #print("NO TUMOR")
        data={
        "disease":"Healthy Brain",
        "abstraction":"Atopic dermatitis(Eczema) is a condition that causes dry, itchy and inflamed skin. It's common in young children but can occur at any age. Atopic dermatitis is long lasting(Chronic) and tends to flare sometimes. It can be irritating but it's not contagious. <br> People with atopic dermatites are at risk of diveloping food allergies, high fever and asthma. <br> Moisturizing regularly and following other skin care habits can relieve itching and prevent new outbreaks(flares). Treatment may also include medicated ointments or cream.",
        "symptoms_define":"In infants, the itchy rash can lead to an oozing, crusting condition, mainly on the face and scalp. It can also happen on their arms, legs, back, and chest. Newborn babies can show symptoms within the first few weeks or months after birth. The rash usually happens on your face, the backs of your knees, wrists, hands, or feet. Your skin will probably be very dry, thick, or scaly. In fair-skinned people, these areas may start out reddish and then turn brown. In darker-skinned people, eczema can affect skin pigments, making the affected area lighter or darker. Developing a basic skin care routine may help prevent eczema flares. The following tips may help reduce the drying effects of bathing:",
        "symptoms_list":"<li>Dry, cracked skin</li><li>Itchiness (pruritus)</li><li>Rash on swollen skin that varies in color depending on your skin color</li><li>Oozing and crusting, Thickened skin</li><li>Small, raised bumps, on brown or Black skin</li><li>Darkening of the skin around the eyes Raw, sensitive skin from scratching</li>",
        "treatment":"<li>Moisturize your skin at least twice a day</li><li>Take a daily bath or Shower</li><li>Use a gentle, nonsoap cleaner</li>"
    }
        return data
#-----------------------------------------------------------------------------------------------------------------    
def lungs_models_prediction(lungs_Images):
    test_image=load_img(lungs_Images, target_size=(224, 224))
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
    result = lungs_model.predict(test_image).round(3) 
    #print('@@ Raw result = ', result)
    pred=np.argmax(result)
    if pred == 0:
        data={
            "disease":"Bacterial Pneumonia",
            "abstraction":"Hormonal imbalances, especially in androgens, can contribute to acne. Testing hormone levels, particularly in cases of adult-onset acne or in females with signs of hormonal imbalance, may be considered.Skin Biopsy: In rare cases, a skin biopsy may be performed to examine a small piece of skin under a microscope. This can help confirm the diagnosis and rule out other skin conditions. Patch Testing: Patch testing may be done to identify any allergic reactions to topical medications or skincare products, which could exacerbate or contribute to acne. Comedone Extraction: Dermatologists may perform comedone extraction to remove blackheads and whiteheads. This is a manual extraction process that should be done by a trained professional. Imaging Techniques: Imaging techniques such as ultrasound or high-frequency ultrasound may be used to assess the depth and severity of acne lesions.",
            "symptoms_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age.",
            "symptoms_list":"<li><strong>Comedones:</strong> These are non-inflammatory lesions and can be open (blackheads) or closed (whiteheads).</li><li><strong>Papules:</strong> Small, red, raised bumps that may be tender to the touch.</li><li><strong>Pustules:</strong> Pimples filled with pus. They are red at the base and have a white or yellow center.</li><li><strong>Nodules:</strong> Large, painful, solid lesions located deep within the skin.</li><li><strong>Cysts:</strong> Deep, painful, pus-filled lumps that can cause scarring.</li>",
            "treatment":"The goal of acne treatment is to reduce oil production, prevent clogged pores, and manage inflammation. Treatment options include:<br><strong>Topical Treatments:</strong><br><li><strong>Benzoyl peroxide:</strong> Kills bacteria and removes excess oil. Retinoids: Unclog pores and promote the exfoliation of dead skin cells. Topical antibiotics: Reduce bacteria on the skin.</li><li><strong>Oral medications:</strong> <li><strong>Antibiotics:</strong> Oral antibiotics may be prescribed for moderate to severe acne to reduce inflammation and bacteria. certain oral contraceptives can help regulate hormones and reduce acne</li>. <li><strong>Isotretinoin (Accutane):</strong> A powerful oral medication for severe acne. It is usually reserved for cases that haven't responded to other treatments due to potential side effects.</li></li><li><strong>Light and laser therapy:</strong> Certain light-based therapies can target bacteria and reduce inflammation.</li><li><strong>Chemical peels:</strong> Exfoliate the skin, helping to unclog pores and improve the appearance of acne.</li>"
        }
        #print("bacterial pneumonia")
        return data
    elif pred==1:
        #print("normal")
        data={
            "disease":"Normal",
            "abstraction":"Hormonal imbalances, especially in androgens, can contribute to acne. Testing hormone levels, particularly in cases of adult-onset acne or in females with signs of hormonal imbalance, may be considered.Skin Biopsy: In rare cases, a skin biopsy may be performed to examine a small piece of skin under a microscope. This can help confirm the diagnosis and rule out other skin conditions. Patch Testing: Patch testing may be done to identify any allergic reactions to topical medications or skincare products, which could exacerbate or contribute to acne. Comedone Extraction: Dermatologists may perform comedone extraction to remove blackheads and whiteheads. This is a manual extraction process that should be done by a trained professional. Imaging Techniques: Imaging techniques such as ultrasound or high-frequency ultrasound may be used to assess the depth and severity of acne lesions.",
            "symptoms_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age.",
            "symptoms_list":"<li><strong>Comedones:</strong> These are non-inflammatory lesions and can be open (blackheads) or closed (whiteheads).</li><li><strong>Papules:</strong> Small, red, raised bumps that may be tender to the touch.</li><li><strong>Pustules:</strong> Pimples filled with pus. They are red at the base and have a white or yellow center.</li><li><strong>Nodules:</strong> Large, painful, solid lesions located deep within the skin.</li><li><strong>Cysts:</strong> Deep, painful, pus-filled lumps that can cause scarring.</li>",
            "treatment":"The goal of acne treatment is to reduce oil production, prevent clogged pores, and manage inflammation. Treatment options include:<br><strong>Topical Treatments:</strong><br><li><strong>Benzoyl peroxide:</strong> Kills bacteria and removes excess oil. Retinoids: Unclog pores and promote the exfoliation of dead skin cells. Topical antibiotics: Reduce bacteria on the skin.</li><li><strong>Oral medications:</strong> <li><strong>Antibiotics:</strong> Oral antibiotics may be prescribed for moderate to severe acne to reduce inflammation and bacteria. certain oral contraceptives can help regulate hormones and reduce acne</li>. <li><strong>Isotretinoin (Accutane):</strong> A powerful oral medication for severe acne. It is usually reserved for cases that haven't responded to other treatments due to potential side effects.</li></li><li><strong>Light and laser therapy:</strong> Certain light-based therapies can target bacteria and reduce inflammation.</li><li><strong>Chemical peels:</strong> Exfoliate the skin, helping to unclog pores and improve the appearance of acne.</li>"
        }
        return data
    elif pred==2:
        #print("tuberculosis")
        data={
            "disease":"Tuberculosis",
            "abstraction":"Hormonal imbalances, especially in androgens, can contribute to acne. Testing hormone levels, particularly in cases of adult-onset acne or in females with signs of hormonal imbalance, may be considered.Skin Biopsy: In rare cases, a skin biopsy may be performed to examine a small piece of skin under a microscope. This can help confirm the diagnosis and rule out other skin conditions. Patch Testing: Patch testing may be done to identify any allergic reactions to topical medications or skincare products, which could exacerbate or contribute to acne. Comedone Extraction: Dermatologists may perform comedone extraction to remove blackheads and whiteheads. This is a manual extraction process that should be done by a trained professional. Imaging Techniques: Imaging techniques such as ultrasound or high-frequency ultrasound may be used to assess the depth and severity of acne lesions.",
            "symptoms_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age.",
            "symptoms_list":"<li><strong>Comedones:</strong> These are non-inflammatory lesions and can be open (blackheads) or closed (whiteheads).</li><li><strong>Papules:</strong> Small, red, raised bumps that may be tender to the touch.</li><li><strong>Pustules:</strong> Pimples filled with pus. They are red at the base and have a white or yellow center.</li><li><strong>Nodules:</strong> Large, painful, solid lesions located deep within the skin.</li><li><strong>Cysts:</strong> Deep, painful, pus-filled lumps that can cause scarring.</li>",
            "treatment":"The goal of acne treatment is to reduce oil production, prevent clogged pores, and manage inflammation. Treatment options include:<br><strong>Topical Treatments:</strong><br><li><strong>Benzoyl peroxide:</strong> Kills bacteria and removes excess oil. Retinoids: Unclog pores and promote the exfoliation of dead skin cells. Topical antibiotics: Reduce bacteria on the skin.</li><li><strong>Oral medications:</strong> <li><strong>Antibiotics:</strong> Oral antibiotics may be prescribed for moderate to severe acne to reduce inflammation and bacteria. certain oral contraceptives can help regulate hormones and reduce acne</li>. <li><strong>Isotretinoin (Accutane):</strong> A powerful oral medication for severe acne. It is usually reserved for cases that haven't responded to other treatments due to potential side effects.</li></li><li><strong>Light and laser therapy:</strong> Certain light-based therapies can target bacteria and reduce inflammation.</li><li><strong>Chemical peels:</strong> Exfoliate the skin, helping to unclog pores and improve the appearance of acne.</li>"
        }
        return data
    elif pred==3:
        #print("viral pneumonia")
        data={
            "disease":"Viral Pneumonia",
            "abstraction":"Hormonal imbalances, especially in androgens, can contribute to acne. Testing hormone levels, particularly in cases of adult-onset acne or in females with signs of hormonal imbalance, may be considered.Skin Biopsy: In rare cases, a skin biopsy may be performed to examine a small piece of skin under a microscope. This can help confirm the diagnosis and rule out other skin conditions. Patch Testing: Patch testing may be done to identify any allergic reactions to topical medications or skincare products, which could exacerbate or contribute to acne. Comedone Extraction: Dermatologists may perform comedone extraction to remove blackheads and whiteheads. This is a manual extraction process that should be done by a trained professional. Imaging Techniques: Imaging techniques such as ultrasound or high-frequency ultrasound may be used to assess the depth and severity of acne lesions.",
            "symptoms_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age.",
            "symptoms_list":"<li><strong>Comedones:</strong> These are non-inflammatory lesions and can be open (blackheads) or closed (whiteheads).</li><li><strong>Papules:</strong> Small, red, raised bumps that may be tender to the touch.</li><li><strong>Pustules:</strong> Pimples filled with pus. They are red at the base and have a white or yellow center.</li><li><strong>Nodules:</strong> Large, painful, solid lesions located deep within the skin.</li><li><strong>Cysts:</strong> Deep, painful, pus-filled lumps that can cause scarring.</li>",
            "treatment":"The goal of acne treatment is to reduce oil production, prevent clogged pores, and manage inflammation. Treatment options include:<br><strong>Topical Treatments:</strong><br><li><strong>Benzoyl peroxide:</strong> Kills bacteria and removes excess oil. Retinoids: Unclog pores and promote the exfoliation of dead skin cells. Topical antibiotics: Reduce bacteria on the skin.</li><li><strong>Oral medications:</strong> <li><strong>Antibiotics:</strong> Oral antibiotics may be prescribed for moderate to severe acne to reduce inflammation and bacteria. certain oral contraceptives can help regulate hormones and reduce acne</li>. <li><strong>Isotretinoin (Accutane):</strong> A powerful oral medication for severe acne. It is usually reserved for cases that haven't responded to other treatments due to potential side effects.</li></li><li><strong>Light and laser therapy:</strong> Certain light-based therapies can target bacteria and reduce inflammation.</li><li><strong>Chemical peels:</strong> Exfoliate the skin, helping to unclog pores and improve the appearance of acne.</li>"
        }
        return data
#----------------------------------------------------------------------------------------------------------
def skin_models_prediction(skin_Images):
    test_image=load_img(skin_Images, target_size=(224, 224))
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
    result = skin_model.predict(test_image).round(3) # predict diseased palnt or not
    #print('@@ Raw result = ', result)
    pred=np.argmax(result)
    if pred == 0:
        #print("acne")
        data={
            "disease":"Acne on Skin",
            "abstraction":"Hormonal imbalances, especially in androgens, can contribute to acne. Testing hormone levels, particularly in cases of adult-onset acne or in females with signs of hormonal imbalance, may be considered.Skin Biopsy: In rare cases, a skin biopsy may be performed to examine a small piece of skin under a microscope. This can help confirm the diagnosis and rule out other skin conditions. Patch Testing: Patch testing may be done to identify any allergic reactions to topical medications or skincare products, which could exacerbate or contribute to acne. Comedone Extraction: Dermatologists may perform comedone extraction to remove blackheads and whiteheads. This is a manual extraction process that should be done by a trained professional. Imaging Techniques: Imaging techniques such as ultrasound or high-frequency ultrasound may be used to assess the depth and severity of acne lesions.",
            "symptoms_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age.",
            "symptoms_list":"<li><strong>Comedones:</strong> These are non-inflammatory lesions and can be open (blackheads) or closed (whiteheads).</li><li><strong>Papules:</strong> Small, red, raised bumps that may be tender to the touch.</li><li><strong>Pustules:</strong> Pimples filled with pus. They are red at the base and have a white or yellow center.</li><li><strong>Nodules:</strong> Large, painful, solid lesions located deep within the skin.</li><li><strong>Cysts:</strong> Deep, painful, pus-filled lumps that can cause scarring.</li>",
            "treatment":"The goal of acne treatment is to reduce oil production, prevent clogged pores, and manage inflammation. Treatment options include:<br><strong>Topical Treatments:</strong><br><li><strong>Benzoyl peroxide:</strong> Kills bacteria and removes excess oil. Retinoids: Unclog pores and promote the exfoliation of dead skin cells. Topical antibiotics: Reduce bacteria on the skin.</li><li><strong>Oral medications:</strong> <li><strong>Antibiotics:</strong> Oral antibiotics may be prescribed for moderate to severe acne to reduce inflammation and bacteria. certain oral contraceptives can help regulate hormones and reduce acne</li>. <li><strong>Isotretinoin (Accutane):</strong> A powerful oral medication for severe acne. It is usually reserved for cases that haven't responded to other treatments due to potential side effects.</li></li><li><strong>Light and laser therapy:</strong> Certain light-based therapies can target bacteria and reduce inflammation.</li><li><strong>Chemical peels:</strong> Exfoliate the skin, helping to unclog pores and improve the appearance of acne.</li>"
        }
        return data # if index 0 burned leaf
    elif pred==1:
        #print("normal")
        data={
        "disease":"Good Skin.",
        "abstraction":"Good skin is typically characterized by its smooth texture, even tone, and healthy glow. It is free of blemishes, dryness, and excessive oiliness, and it reflects an overall sense of vitality and well-being. Here's a more detailed description of the qualities that define good skin. Remember, good skin is not just about genetics or luck; it is also a result of conscious choices, consistent care, and a commitment to overall health and well-being. By adopting healthy habits, incorporating a personalized skincare routine, and addressing any underlying skin concerns, you can nurture your skin's natural beauty and achieve a healthy, glowing complexion.",
        "symptoms_define":"---",
        "symptoms_list":"---",
        "treatment":"<li>Choose gentle skincare products: Avoid harsh soaps and cleansers that can strip away natural oils and irritate the skin.</li><li>Moisturize regularly: Moisturizing helps maintain skin hydration and prevent dryness.</li><li>Exfoliate regularly: Exfoliating removes dead skin cells and promotes skin cell turnover.</li><li>Protect your skin from the sun: Regular use of sunscreen with an SPF of 30 or higher protects against sun damage and premature aging.</li><li>Manage stress effectively: Chronic stress can worsen skin conditions and accelerate aging.</li><li>Avoid smoking: Smoking damages skin cells and accelerates aging.</li><li>Maintain a healthy sleep routine: Adequate sleep allows the skin to repair and regenerate.</li>"
    }
        return data
    elif pred==2:
        #print("vascular tumors")
        data={
        "disease":"Vascular Tumor.",
        "abstraction":"Cutaneous vascular proliferations are a vast and complex spectrum. Many appear as hamartomas in infancy; others are acquired neoplasms. Some vascular proliferations are hyperplastic in nature, although they mimic hemangiomas, i.e., neoplasms. The vast majority of the vascular lesions are hemangiomas. Between the hemangiomas and frankly angiosarcomas, there is a group of neoplasms that are angiosarcomas, albeit ones of low grade histologically and, probably, biologically. The term 'hemangioendothelioma' has been created to encompass these neoplasms. Vascular proliferations are, fundamentally, composed of endothelial cells. Some hemangiomas, however, contain also abundant pericytic, smooth muscle, or interstitial components, or a combination of them. These heterogeneous cellular components are present usually in hemangiomas. Some of the newly described vascular proliferations, however, are difficult to differentiate from some of the angiosarcomas. Others are markers, occasionally, of serious conditions such as Fabry's Disease (angiokeratoma) and POEM's syndrome (glomeruloid hemangioma). Kaposi's sarcoma continues to be an enigma. The demonstration of Herpes virus 8 in this condition raises doubt about its neoplastic nature. The demonstration of endothelial differentiation of its nodular lesions is tenuous and its true nature remains unresolved. While physicians have known about post-mastectomy angiosarcomas from the origin of the radical mastectomy, a new group of unusual vascular proliferations of the mammary skin are being defined. These lesions arise in the setting of breast-conserving surgical treatment with adjuvant radiation therapy. The incubation period is usually 3 to 5 years, in contrast with the 10, or more, in classical cases of post-mastectomy angiosarcoma. These lesions usually are subtle, both clinically and histologically, in contrast with the 'classical,' dramatic presentation of mammary angiosarcoma. The spectrum of findings ranges from 'simple' lymphangiectasia-like vascular proliferations to unequivocal angiosarcomas. The pathogenesis of these lesions remains a mystery. There are very few clues that allow one to separate hemangiomas from angiosarcomas. The presence of heterologous cellular elements and, particularly, well-developed smooth muscle components tends to favor a hemangioma. Similarly, the presence of thrombosis usually supports hemangioma. Nevertheless, there are no unequivocal or reliable individual diagnostic criteria. A thorough knowledge of the different conditions and their differential diagnoses eventually leads to the proper diagnosis in most cases.",
        "symptoms_define":"Clinical experts may use a variety of tests and diagnostic tools to evaluate vascular tumors. The specific tests conducted can depend on the suspected type of vascular tumor and the patient's symptoms.",
        "symptoms_list":"<li><strong>Pain:</strong> Vascular tumors may cause pain in the affected area. The pain can range from mild to severe and may be constant or intermittent.</li><li><strong>Swelling:</strong>Tumors involving blood vessels can lead to swelling in the affected region. The extent of swelling depends on the size and location of the tumor.</li><li><strong>Bruising:</strong> Easy bruising or the development of unusual bruises may be a symptom of certain vascular tumors.</li><li><strong>Bleeding:</strong> Vascular tumors may be associated with bleeding, either externally or internally. This can manifest as visible bleeding, such as from the skin, or internal bleeding leading to anemia.</li><li><strong>Skin changes:</strong> Changes in skin color or texture overlying the tumor may occur. This can include redness, bluish discoloration, or a warm feeling in the affected area.</li><li><strong>Functional impairment:</strong> Depending on the location and size of the tumor, there may be functional impairment of nearby organs or structures. For example, a vascular tumor in the brain may cause neurological symptoms.</li>",
        "treatment":"<li>Observation and Monitoring:<ul>Small, asymptomatic vascular tumors may not require immediate treatment. In such cases, the healthcare provider might opt for a watch-and-wait approach, monitoring the tumor for any changes.</ul></li><li><strong>Medications:</strong><ul><strong>Corticosteroids:</strong> For certain benign vascular tumors like infantile hemangiomas, corticosteroids may be prescribed to help reduce the size of the tumor.<ul><strong>Propranolol:</strong></ul> This beta-blocker has been found to be effective in treating infantile hemangiomas, especially those that are proliferating rapidly.</ul></li><li><strong>Surgery:</strong>Surgical removal may be considered for some vascular tumors, particularly if they are causing symptoms, growing rapidly, or if there's a concern about malignancy. For benign tumors like hemangiomas, surgery may be performed to remove the tumor and, in some cases, reconstruct affected tissues.</li><li><strong>Radiation Therapy:</strong><ul>Radiation therapy may be used in the treatment of some vascular tumors, particularly malignant ones like angiosarcomas. It may be used as a primary treatment or in conjunction with surgery.</ul></li><li><strong>Embolization:</strong><ul>Embolization is a procedure in which substances are injected into blood vessels to block or reduce blood flow to the tumor. This may be used for certain types of vascular tumors, particularly if they are difficult to reach surgically.</ul></li><li><strong>Chemotherapy:</strong>Chemotherapy may be considered for certain vascular tumors, especially if they are malignant and have spread to other parts of the body.</li>"
    }
        return data
    elif pred==3:
        #print("fungal")
        data={
        "disease":"Fungal Skin.",
        "abstraction":"The treatment of fungal diseases is typically carried out by medical professionals, such as doctors or healthcare providers. The specific approach to treatment depends on the type of fungal infection, its severity, and the patient's overall health. Common fungal infections include skin infections (such as ringworm or athlete's foot), nail infections, and systemic infections.Topical Antifungals: For localized infections on the skin or nails, topical antifungal creams, ointments, or powders are often prescribed. Oral Antifungals: Systemic or widespread fungal infections may require oral antifungal medications. These medications are absorbed into the bloodstream and can reach affected areas throughout the body. In cases of fungal infections on the scalp or other hair-bearing areas, antifungal shampoos or washes may be recommended.",
        "symptoms_define":"Fungal diseases can affect various parts of the body and manifest in a wide range of symptoms. The specific symptoms depend on the type of fungus involved and the part of the body affected. Here are some common symptoms associated with fungal diseases:",
        "symptoms_list":"<li><strong>Skin Infections:</strong><ul><strong>Ringworm (Tinea corporis):</strong>Red, itchy, and circular rashes on the skin.</ul><ul><strong>Athlete's Foot (Tinea pedis):</strong>Itchy, red, and peeling skin, often between the toes.</ul><ul><strong>Jock Itch (Tinea cruris):</strong>Itchy, red rash in the groin area.</ul></li><li><strong>Nail Infections:</strong><ul><strong>Onychomycosis:</strong>Thickened, discolored, and brittle nails. Nails may become yellow or white.</ul></li><li><strong>Oral Infection:</strong><ul><strong>Oral Thrust: </strong>White patches on the tongue, onner cheeks, and roof of the mouth. Soreness and difficulty swallowing may also occur.</ul></li><li><strong>Respiratory Infections:</strong><ul><strong>Aspergillosis:</strong>Cough, wheezing, chest pain, and shortness of breath.</ul><ul><strong>Pneumocystis pneumonia (PCP): </strong>Fever, cough, and difficulty breathing, particularly in individuals with weakened immune systems.</ul></li><li><strong>Systemic Infections:</strong><ul><strong>Candidemia:</strong>Fever and chills. In severe cases, it can lead to organ failure.</ul><ul><strong>Histoplasmosis: </strong>Fever, cough, fatigue, and sometimes joint pain. It can affect the lungs and other organs.</ul></li>",
        "treatment":"<h3>Clinical experts use a variety of tests to diagnose fungal diseases. The specific tests conducted depend on the suspected fungal infection and the patient's symptoms. Some common fungal disease tests include:</h3><li><strong>Microscopic Examination:</strong><ul><strong>Potassium Hydroxide (KOH) Preparation:</strong>This test involves placing a sample of the affected tissue or fluid in a solution of potassium hydroxide. The solution breaks down cells, leaving fungal elements more visible under a microscope.</ul></li><li>Fungal Culture: A sample of the infected tissue or fluid is cultured on a special medium to encourage the growth of fungi. This helps identify the specific type of fungus causing the infection.</li><li>Antibody Detection: Some fungal infections trigger the production of specific antibodies, which can be detected in the blood. Enzyme-linked immunosorbent assay (ELISA) is a common method for this purpose.</li>"
    }
        return data
#------------------------------------------------------------------------------------------------------------
app=Flask(__name__)
app.secret_key="secret key"
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")
@app.route("/diganosis", methods=['GET'])
def diagnosis():
    return render_template("diganosis.html")
@app.route("/credits", methods=["GET"])
def credits():
    return render_template("credit.html")
@app.route("/help", methods=["GET"])
def help():
    return render_template("help.html")
@app.route("/analysis", methods=["GET"])
def analysis():
    return render_template("analysis.html")
@app.route("/upload/<disease>", methods=['GET','POST'])
def upload(disease):
    if request.method=="POST":
        if disease=="lung":
            return render_template(upload,disease=disease)
        elif disease=="brain":
            return render_template(upload, disease=disease)
        elif disease=="skin":
            return render_template(upload,disease=disease)
    return render_template("upload.html", disease=disease)
@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    if request.method=='GET':
        return render_template("diabetesform.html")
    elif request.method=='POST':
        # data=request.form
        age = request.form['age']
        hypertension= request.form['hypertension']
        hba1c = request.form['hba1c']
        gendervalue = request.form['gender']
        if(gendervalue=="Female"):
            gender=0
        else:
            gender=1
        smokingHistory = request.form['smokingHistory']
        # if(smokingHistory=="never"):
        #     smoke=4
        # elif(smokingHistory=="current"):
        #     smoke=1
        # elif(smokingHistory=="formal"):
        #     smoke=3
        # else:
        #     smoke=0
        heartDisease = request.form['heartDisease']
        # if(heartDisease=='no'):
        #     heartDisease=0
        # else:
        #     heartDisease=1
        weight = int(request.form['weight'])
        bloodGlucose=int(request.form['bloodGlucose'])
        bmi=request.form['BMI']
        # to_predict = np.array([gender, age, hypertension, heartDisease, smoke, bmi, hba1c, bloodGlucose])
        to_predict = [[gender, age, hypertension, heartDisease, smokingHistory, bmi, hba1c, bloodGlucose]]
        result = diabetes_model.predict(X=to_predict)
        #print(result)
        if(result==0):
            #print("No diabetes")
            file_url="static/assest/diabetes result.webp"
            data={
            "disease":"No Risk of Diabetes",
            "abstraction":"In people without diabetes, achieving target blood sugar levels has several key health benefits: Helps prevent weight gain, or achieve weight loss goals. Reduces the risk of insulin resistance and type 2 diabetes. Reduced stress hormones and inflammation.",
            "symptoms_define":"",
            "symptoms_list":"",
            "treatment":""
            }
            return render_template(result, data=data, user_image=file_url)
        else:
            #print("diabetes")
            file_url="static/assest/diabetes result.webp"
            data={
            "disease":"Risk of Diabetes",
            "abstraction":"Diabetes mellitus, especially type 2 diabetes, is an epidemic requiring global attention as a cardiovascular disease (CVD) risk. In addition to well-known microvascular complications such as retinopathy or nephropathy, diabetes confers the substantial burden of CVD morbidity and mortality through macrovascular complications even in early- or pre-stages. Because of its asymptomatic onset and progression, population-based screening is essential for early detection of diabetes mellitus before the development of vascular complications, including CVD. Many modifiable risk factors such as hyperglycemia, hypertension, or dyslipidemia must be adequately and simultaneously controlled for prevention of CVDs in people with established diabetes mellitus.",
            "symptoms_define":"Diabetes symptoms depend on how high your blood sugar is. Some people, especially if they have prediabetes, gestational diabetes or type 2 diabetes, may not have symptoms. In type 1 diabetes, symptoms tend to come on quickly and be more severe.",
            "symptoms_list":"<li>Urinate (pee) a lot, often at night</li><li>Lose Weight without trying</li><li>Have Blurry Vision</li><li>Have numb or tingling hands or feet</li><li>have a very dry skin(Abnormally)</li>",
            "treatment":"Type 1 diabetes can't be prevented. But the healthy lifestyle choices that help treat prediabetes, type 2 diabetes and gestational diabetes can also help prevent them:<br><li>Eat healthy foods.</li><li>Lose excess Weight.</li><li>Get involved in Physical Activity.</li>"
            }
            return render_template(result, data=data, user_image=file_url)
        return render_template("diabetesform.html")

@app.route("/heart", methods=['GET','POST'])
def heart():
    if request.method=='GET':
        return render_template("heartform.html")
    elif request.method=='POST':
        Age=int(request.form['age'])
        sex =request.form['gender']
        if sex=='male':
            sexno=1
        elif sex=='female':
            sexno=0
        else:
            sexno=1
        chestPainType =int(request.form['chestpain'])
        RestingBP = int(request.form['BP'])
        cholesterol =int(request.form['Cholesterol'])
        FastingBS=int(request.form['bloodsugar'])
        RestingECG=int(request.form['ECG'])
        MaxHR =int(request.form['MaxHR'])
        ExerciseAngina=int(request.form['ExerciseAngina'])
        Oldpeak=float(request.form['OldPeak'])
        ST_Slope=int(request.form['stslope'])
    
        # Include Sex in the input data
        input_data = [[Age, sexno, chestPainType, RestingBP, cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]]
        # Make predictions
        prediction = heart_model.predict(input_data)
        # Display the result
        if prediction == 0:
            #print("No heart disease")
            file_url="static/assest/Healthy_heart.jpg"
            data={
            "disease":"Normal Report",
            "abstraction":"A healthy heart is the cornerstone of overall well-being, playing a pivotal role in sustaining life and vitality. The term 'healthy heart' encompasses a state where the heart functions optimally, efficiently pumping blood throughout the body and supplying vital nutrients and oxygen to every cell. Achieving and maintaining a healthy heart involves a multifaceted approach, integrating lifestyle choices, dietary habits, and regular physical activity.",
            "symptoms_define":"Here some ways to tell if your heart is healthy — now and in the future.",
            "symptoms_list":"<li>Controlled Blood Pressure. </li><li>Good Sleep(Around 8 to 9 hours per 24 hours).</li><li>Good Oral Health.</li><li>High Energy Levels.<li>",
            "treatment":""
            }
            return render_template(result, data=data, user_image=file_url)
        else:
            #print("Heart disease")
            file_url="static/assest/Unhealthy_heart.jpg"
            data={
            "disease":"Heart Disease",
            "abstraction":"The diagnosis of heart disease in most cases depends on a complex combination of clinical and pathological data. Because of this complexity, there exists a significant amount of interest among clinical professionals and researchers regarding the efficient and accurate prediction of heart disease. In this paper, we develop a heart disease predict system that can assist medical professionals in predicting heart disease status based on the clinical data of patients. Our approaches include three steps. Firstly, we select 13 important clinical features, i.e., age, sex, chest pain type, trestbps, cholesterol, fasting blood sugar, resting ecg, max heart rate, exercise induced angina, old peak, slope, number of vessels colored, and thal. Secondly, we develop an artificial neural network algorithm for classifying heart disease based on these clinical features. The accuracy of prediction is near 80%. Finally, we develop a user-friendly heart disease predict system (HDPS). The HDPS system will be consisted of multiple features, including input clinical data section, ROC curve display section, and prediction performance display section (execute time, accuracy, sensitivity, specificity, and predict result). Our approaches are effective in predicting the heart disease of a patient. The HDPS system developed in this study is a novel approach that can be used in the classification of heart disease.",
            "symptoms_define":"Coronary artery disease is a common heart condition that affects the major blood vessels that supply the heart muscle. Cholesterol deposits (plaques) in the heart arteries are usually the cause of coronary artery disease. The buildup of these plaques is called atherosclerosis (ath-ur-o-skluh-ROE-sis). Atherosclerosis reduces blood flow to the heart and other parts of the body. It can lead to a heart attack, chest pain (angina) or stroke.<br> Coronary artery disease symptoms may be different for men and women. For instance, men are more likely to have chest pain. Women are more likely to have other symptoms along with chest discomfort, such as shortness of breath, nausea and extreme fatigue.",
            "symptoms_list":"<li>Chest pain, chest tightness, chest pressure and chest discomfort (angina)</li><li>Shortness of breath</li><li>Pain in the neck, jaw, throat, upper belly area or back</li><li>Pain, numbness, weakness or coldness in the legs or arms if the blood vessels in those body areas are narrowed</li>",
            "treatment":"The same lifestyle changes used to manage heart disease may also help prevent it. Try these heart-healthy tips:<br><li>Don't smoke.</li><li>Eat a diet that's low in salt and saturated fat.</li><li>Exercise at least 30 minutes a day on most days of the week.</li><li>Maintain a healthy weight.</li><li>Reduce and manage stress.</li><li>Control high blood pressure, high cholesterol and diabetes.</li><li>Get good sleep. Adults should aim for 7 to 9 hours daily.</li>"
            }
            return render_template(result, data=data, user_image=file_url)
    
#-----------------------------------------------------------------------------------------------------------
@app.route("/predict/<disease>", methods=['POST'])
def predict(disease):
    if request.method=='POST':
        if disease=='lungs':
            #print("lungs function is working")
            file=request.files['image']
            filename=file.filename
            if filename=="":
                flash(flash_msz, "error")
                return render_template('index.html')
            file_path = 'userUpload/' + filename
            file_path_full = 'static/' + file_path
            file.save(file_path_full)
            data=lungs_models_prediction(lungs_Images=file_path_full)
            file_url = url_for('static', filename=file_path)
            return render_template(result, data=data, user_image=file_url)
        elif disease == 'brain':
            file=request.files['image']
            filename=file.filename
            #print(filename)
            if filename=="":
                flash(flash_msz, "error")
                return render_template('index.html')
            file_path = 'userUpload/' + filename
            file_path_full = 'static/' + file_path
            file.save(file_path_full)
            data=brain_models_prediction(brain_Images=file_path_full)
            file_url = url_for('static', filename=file_path)
            return render_template(result, data=data, user_image=file_url)
        elif disease=='skin':
            file=request.files['image']
            filename=file.filename
            if filename=="":
                flash(flash_msz, "error")
                
                return render_template('index.html')
            file_path = 'userUpload/' + filename
            file_path_full = 'static/' + file_path
            file.save(file_path_full)
            data=skin_models_prediction( skin_Images=file_path_full)
            file_url = url_for('static', filename=file_path)
            return render_template(result, data=data, user_image=file_url)
#---------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    index='index.html'
    credit='credit.html'
    help='help.html'
    upload='upload.html'
    flash_msz="Please Enter the Images. Image is not Inserted!!"
    app.run(debug=True)
