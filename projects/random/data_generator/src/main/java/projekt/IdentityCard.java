package projekt;

import java.util.Random;

public class IdentityCard {

	private String identityCardNumber="";

	public IdentityCard() {
		Random random = new Random();
		int[] Weights = new int[] { 7,3,1};
		int SumWithWeights=0;

		for(int i=0;i<3;i++) {
			char c = (char)(random.nextInt(26) + 'A');
			this.identityCardNumber+= c;
			SumWithWeights+=Weights[i]*(Integer.valueOf(c)-55);
		}

		int[] Weights2 = new int[] { 7,3,1,7,3};
		while(true) {
			String tmp="";
			int SumWithWeights2 = SumWithWeights;
			for(int i=0;i<5;i++) {
				int num = random.nextInt(10);
				tmp+=String.valueOf(num);
				SumWithWeights2+=Weights2[i]*num;
			}
			int CheckNumber = SumWithWeights2%10;
			if((SumWithWeights2+CheckNumber*9)%10==0){
				this.identityCardNumber+=String.valueOf(CheckNumber);
				this.identityCardNumber+=String.valueOf(tmp);
				break;
			}
		}
	}

	public String getIdentityCardNumber() {
		return identityCardNumber;
	}
}
