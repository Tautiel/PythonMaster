# CREO LA SCHEDA DELLA MIA MOTO

modello = "Beta RR Motard"
anno = 2023
cilindrata = 125
prz_eur = 3500.00
in_garanzia = True
print(f"La mia moto {modello} di cilindrata {cilindrata} del {anno} pagata {prz_eur} euro ")

# CONSUMO

km_percorsi = 187
litri_usati = 5.2

print(f"I Km per litro percorsi sono {km_percorsi/litri_usati:.2f}")

# CALCOLA TAGLIANDO

km_attuali = input("Km attuali? ")
km_attuali = int(km_attuali)
km_tagliando = input("Km ultimo tagliando? ")
km_tagliando = int(km_tagliando)
km_rimanenti = km_attuali - km_tagliando
km_pertagliando = 4000
km_mancanti = km_pertagliando - km_rimanenti
print(f" I Km che mancano al prossimo tagliando sono {km_mancanti} ")

