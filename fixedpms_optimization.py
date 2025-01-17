import os
import ewmlib
import pickle


if __name__ == "__main__":
    
    os.makedirs("./results", exist_ok=True)
    
    # EqODE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print('\nEqODE (classical)')
    kappa, B, Aplus, yp_ref, results = ewmlib.opti_fixedpms_eqode("classical")
    with open("./results/fixedpms_eqode_classical.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'Aplus':Aplus,
            'yp_ref':yp_ref,
            'results':results
            }, file)

    print('\nEqODE (highre)')
    kappa, B, Aplus, yp_ref, results = ewmlib.opti_fixedpms_eqode("highre")
    with open("./results/fixedpms_eqode_highre.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'Aplus':Aplus,
            'yp_ref':yp_ref,
            'results':results
            }, file)






    # Reichardt fixed B1 B2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print('\nReichardt fixed B1 B2 (classical)')
    
    kappa, B, C, B1, B2, yp_ref, results = ewmlib.opti_fixedpms_reichardt_fixedB1B2("classical")
    with open("./results/fixedpms_reichardt-fixedB1B2_classical.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'C':C,
            'B1':B1,
            'B2':B2,
            'yp_ref':yp_ref,
            'results':results
            }, file)

    print('\nReichardt fixed B1 B2 (highre)')
    kappa, B, C, B1, B2, yp_ref, results = ewmlib.opti_fixedpms_reichardt_fixedB1B2("highre")
    with open("./results/fixedpms_reichardt-fixedB1B2_highre.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'C':C,
            'B1':B1,
            'B2':B2,
            'yp_ref':yp_ref,
            'results':results
            }, file)






    # Spalding <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print('\nSpalding (classical)')
    kappa, B, up_ref, results = ewmlib.opti_fixedpms_spalding("classical")
    with open("./results/fixedpms_spalding_classical.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'up_ref':up_ref,
            'results':results
            }, file)
    
    print('\nSpalding (highre)')
    kappa, B, up_ref, results = ewmlib.opti_fixedpms_spalding("highre")
    with open("./results/fixedpms_spalding_highre.pkl", "wb") as file:
        pickle.dump({
            'kappa':kappa,
            'B':B,
            'up_ref':up_ref,
            'results':results
            }, file)

    